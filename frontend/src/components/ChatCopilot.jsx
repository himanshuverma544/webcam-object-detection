import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  FiCamera,
  FiCheckCircle,
  FiCornerDownLeft,
  FiImage,
  FiMaximize2,
  FiMinimize2,
  FiTrash2,
  FiX,
} from "react-icons/fi";
import { ObjectBoxModal } from "./ObjectBoxModal";
import { sendChatMessage } from "../services/api";

const QUICK_SUGGESTIONS = [
  { label: "📹 Start Camera", prompt: "Start the camera" },
  { label: "👤 Person & Phone", prompt: "Only detect person and cell phone" },
  { label: "🛡️ Guard Scene", prompt: "Guard the scene now" },
  { label: "🎯 70% Confidence", prompt: "Set confidence to 70%" },
  { label: "📸 Snapshot Item", prompt: "Capture snapshot and save as 'My Mug'" },
  { label: "📊 Show Metrics", prompt: "Show session metrics summary" },
];

/**
 * Helper to extract an item name if already present in user prompt.
 */
function extractItemName(text) {
  if (!text) return "";
  const match = text.match(/['"]([^'"]+)['"]/);
  if (match) return match[1].trim();
  const patternMatch = text.match(
    /\b(?:custom item|save as|named|called|item)\s+([a-zA-Z0-9 _-]+)/i
  );
  return patternMatch ? patternMatch[1].trim() : "";
}

export function ChatCopilot({
  cameraStatus,
  isDetecting,
  confidence,
  similarityThreshold = 0.65,
  selectedClasses,
  customItems,
  selectedCustomIds,
  isGuarded,
  hasMetrics = false,
  supportedClasses,
  isMetricsModalOpen = false,
  metricsModalTab = "metrics",
  onExecuteActions,
  onCaptureLiveSnapshot,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content:
        "👋 **Hello! I'm your Vision AI Copilot.**\n\nI can control your camera, toggle COCO classes, adjust sliders, guard the scene, and register custom items via webcam snapshots or uploaded photos.\n\n*How can I help you right now?*",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [inputPrompt, setInputPrompt] = useState("");
  const [attachedFiles, setAttachedFiles] = useState([]); // [{ id, file, previewUrl, objectUrlToRevoke, name }]
  const [cropModalState, setCropModalState] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (isOpen && !isMinimized) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen, isMinimized, isLoading]);

  // Clean up object URLs on unmount
  useEffect(() => {
    return () => {
      attachedFiles.forEach((it) => {
        if (it.objectUrlToRevoke) {
          URL.revokeObjectURL(it.objectUrlToRevoke);
        }
      });
    };
  }, [attachedFiles]);

  // Reset unread count when opening
  useEffect(() => {
    if (isOpen) {
      setUnreadCount(0);
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  const handleToggleOpen = () => {
    setIsOpen((prev) => !prev);
    setIsMinimized(false);
  };

  const handleClearHistory = () => {
    setMessages([
      {
        id: "cleared_" + Date.now(),
        role: "assistant",
        content: "Conversation history cleared. Ready for your next command!",
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const existingNameFromPrompt = extractItemName(inputPrompt);
    const existingName = existingNameFromPrompt || attachedFiles[0]?.name || null;
    const cleanName = existingName || file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ").trim();
    const url = URL.createObjectURL(file);
    setCropModalState({
      items: [{ file, previewUrl: url }],
      defaultName: cleanName || "Custom Item",
      isLiveSnapshot: false,
    });
  };

  const handleLiveSnapshot = async () => {
    if (!onCaptureLiveSnapshot || cameraStatus !== "active") {
      alert("Please start the webcam first before taking a live snapshot!");
      return;
    }
    try {
      const file = await onCaptureLiveSnapshot("live_snapshot");
      if (file) {
        const existingNameFromPrompt = extractItemName(inputPrompt);
        const existingName = existingNameFromPrompt || attachedFiles[0]?.name || null;
        const url = URL.createObjectURL(file);
        setCropModalState({
          items: [{ file, previewUrl: url }],
          defaultName: existingName || "My Item",
          isLiveSnapshot: true,
        });
      }
    } catch (err) {
      console.error("Failed to capture snapshot:", err);
    }
  };

  const handleRemoveAttached = (idToRemove) => {
    setAttachedFiles((prev) => {
      const item = prev.find((it) => it.id === idToRemove);
      if (item?.objectUrlToRevoke) {
        URL.revokeObjectURL(item.objectUrlToRevoke);
      }
      const updated = prev.filter((it) => it.id !== idToRemove);

      const promptTrimmed = inputPrompt.trim();
      const isDefaultPattern =
        !promptTrimmed ||
        /^Save (this|these) (snapshot|photo|\d+ reference photos) as custom item/i.test(
          promptTrimmed
        );

      if (isDefaultPattern) {
        const targetName =
          extractItemName(inputPrompt) || updated[0]?.name || "My Item";
        const count = updated.length;
        if (count === 0) {
          setInputPrompt("");
        } else if (count === 1) {
          setInputPrompt(`Save this photo as custom item '${targetName}'`);
        } else {
          setInputPrompt(
            `Save these ${count} reference photos as custom item '${targetName}'`
          );
        }
      }

      return updated;
    });
  };

  const handleClearAllAttached = () => {
    attachedFiles.forEach((it) => {
      if (it.objectUrlToRevoke) {
        URL.revokeObjectURL(it.objectUrlToRevoke);
      }
    });
    setAttachedFiles([]);
    const promptTrimmed = inputPrompt.trim();
    if (
      /^Save (this|these) (snapshot|photo|\d+ reference photos) as custom item/i.test(
        promptTrimmed
      )
    ) {
      setInputPrompt("");
    }
  };

  const handleSend = async (overridePrompt) => {
    const textToSend = (overridePrompt !== undefined ? overridePrompt : inputPrompt).trim();
    const filesToSend = attachedFiles.map((it) => it.file);
    const primaryFile = filesToSend[0] || null;

    if (!textToSend && filesToSend.length === 0) return;

    const userMessageId = "user_" + Date.now();
    const userMsg = {
      id: userMessageId,
      role: "user",
      content:
        textToSend ||
        (filesToSend.length > 0
          ? `[Uploaded ${filesToSend.length} reference photo${
              filesToSend.length > 1 ? "s" : ""
            }: ${filesToSend.map((f) => f.name).join(", ")}]`
          : ""),
      previewUrls: attachedFiles.map((it) => it.previewUrl),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputPrompt("");
    // Note: Do not clear attachedFiles yet; if Copilot asks a clarification question,
    // we keep the staged files so the user doesn't lose their cropped photos!
    setIsLoading(true);

    // Prepare current UI state context for copilot
    const currentState = {
      cameraStatus,
      isDetecting,
      confidence,
      similarityThreshold,
      selectedClasses,
      customItemCount: customItems.length,
      customItemNames: (customItems || []).map((it) => it.name),
      selectedCustomIds,
      isGuarded,
      hasMetrics: Boolean(hasMetrics),
      hasAttachedImage: filesToSend.length > 0,
      attachedCount: filesToSend.length,
      supportedClassesCount: supportedClasses?.length || 80,
      isMetricsModalOpen: Boolean(isMetricsModalOpen),
      metricsModalTab: metricsModalTab || "metrics",
    };

    // Format recent history for backend
    const historyPayload = messages
      .filter((m) => m.role === "user" || m.role === "assistant")
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      const response = await sendChatMessage({
        message: textToSend,
        history: historyPayload,
        currentState,
        file: primaryFile,
      });

      const assistantMsg = {
        id: "assistant_" + Date.now(),
        role: "assistant",
        content: response.reply,
        actions: response.actions || [],
        action_summaries: response.action_summaries || [],
        used_model: response.used_model,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setMessages((prev) => [...prev, assistantMsg]);

      const hasCustomItemAction =
        Array.isArray(response.actions) &&
        response.actions.some(
          (a) => a.type === "CAPTURE_CUSTOM_ITEM" || a.type === "ADD_CUSTOM_ITEM_IMAGES"
        );

      const isAskingForName =
        typeof response.reply === "string" &&
        (response.reply.toLowerCase().includes("please provide a name") ||
          response.reply.toLowerCase().includes("already exists"));

      // Keep staged files if Copilot is asking for an item name or resolving duplicate name
      if (hasCustomItemAction || !isAskingForName) {
        setAttachedFiles([]);
      }

      // Execute actions in the parent React application
      if (Array.isArray(response.actions) && response.actions.length > 0 && onExecuteActions) {
        onExecuteActions(response.actions, {
          attachedFile: primaryFile,
          attachedFiles: filesToSend,
          onNotifyCopilot: (feedbackText) => {
            if (!feedbackText) return;
            setMessages((prev) => [
              ...prev,
              {
                id: "feedback_" + Date.now(),
                role: "assistant",
                content: feedbackText,
                timestamp: new Date().toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                }),
              },
            ]);
          },
        });
      }

      if (!isOpen) {
        setUnreadCount((c) => c + 1);
      }
    } catch (err) {
      console.error("Chat copilot error:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: "err_" + Date.now(),
          role: "assistant",
          content: `⚠️ **Error communicating with AI Copilot:** ${
            err.message || "Request failed"
          }. Please try again.`,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      {/* Floating launcher trigger button */}
      {!isOpen && (
        <button
          type="button"
          className="copilot-launcher-btn"
          onClick={handleToggleOpen}
          aria-label="Open AI Copilot Chat"
          title="Open AI Copilot (Natural Language Control)"
        >
          <div className="copilot-launcher-inner">
            <span className="copilot-sparkle-icon">✨</span>
            <span className="copilot-launcher-text">AI Copilot</span>
            {unreadCount > 0 && <span className="copilot-unread-badge">{unreadCount}</span>}
          </div>
        </button>
      )}

      {/* Floating Copilot Drawer Panel */}
      {isOpen && (
        <aside
          className={`copilot-drawer ${isMinimized ? "copilot-drawer-minimized" : ""}`}
          aria-label="AI Copilot Interface"
        >
          {/* Header */}
          <header className="copilot-header">
            <div className="copilot-header-info">
              <span className="copilot-header-avatar">✨</span>
              <div>
                <h3 className="copilot-title">Vision AI Copilot</h3>
                <span className="copilot-status-badge">
                  <span className="copilot-status-dot"></span> Ready to command
                </span>
              </div>
            </div>

            <div className="copilot-header-actions">
              <button
                type="button"
                className="copilot-icon-btn"
                onClick={handleClearHistory}
                title="Clear chat history"
              >
                <FiTrash2 size={14} />
              </button>
              <button
                type="button"
                className="copilot-icon-btn"
                onClick={() => setIsMinimized((prev) => !prev)}
                title={isMinimized ? "Expand" : "Minimize"}
              >
                {isMinimized ? <FiMaximize2 size={14} /> : <FiMinimize2 size={14} />}
              </button>
              <button
                type="button"
                className="copilot-icon-btn"
                onClick={handleToggleOpen}
                title="Close Copilot"
              >
                <FiX size={16} />
              </button>
            </div>
          </header>

          {!isMinimized && (
            <>
              {/* Quick suggestions pills */}
              <div className="copilot-suggestions-bar">
                {QUICK_SUGGESTIONS.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className="copilot-suggestion-chip"
                    onClick={() => handleSend(item.prompt)}
                    disabled={isLoading}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              {/* Message scroll container */}
              <div className="copilot-messages-container">
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`copilot-message-row ${
                      m.role === "user" ? "copilot-msg-user" : "copilot-msg-assistant"
                    }`}
                  >
                    {m.role === "assistant" && (
                      <div className="copilot-msg-avatar" aria-hidden="true">
                        ✨
                      </div>
                    )}
                    <div className="copilot-msg-bubble">
                      {Array.isArray(m.previewUrls) && m.previewUrls.length > 0 ? (
                        <div className="copilot-msg-attachments-grid">
                          {m.previewUrls.map((pUrl, pIdx) => (
                            <img key={pIdx} src={pUrl} alt={`Attached ${pIdx + 1}`} />
                          ))}
                        </div>
                      ) : m.previewUrl ? (
                        <div className="copilot-msg-attachment">
                          <img src={m.previewUrl} alt="Attached input" />
                        </div>
                      ) : null}

                      <div className="copilot-msg-markdown">
                        <ReactMarkdown>{m.content}</ReactMarkdown>
                      </div>

                      {/* Executed Action Badges */}
                      {Array.isArray(m.action_summaries) && m.action_summaries.length > 0 && (
                        <div className="copilot-actions-tags">
                          {m.action_summaries.map((summary, sIdx) => (
                            <span key={sIdx} className="copilot-action-badge">
                              <FiCheckCircle size={11} /> {summary}
                            </span>
                          ))}
                        </div>
                      )}

                      <span className="copilot-msg-time">{m.timestamp}</span>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="copilot-message-row copilot-msg-assistant">
                    <div className="copilot-msg-avatar">✨</div>
                    <div className="copilot-msg-bubble copilot-msg-thinking">
                      <span className="copilot-dot-pulse"></span>
                      <span className="copilot-dot-pulse"></span>
                      <span className="copilot-dot-pulse"></span>
                      <span className="copilot-thinking-text">Copilot reasoning...</span>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Multi-Photo Attachment Bar */}
              {attachedFiles.length > 0 && (
                <div className="copilot-multi-attachment-bar">
                  <div className="copilot-multi-attachment-header">
                    <span className="copilot-attachment-count-badge">
                      📸 {attachedFiles.length} reference photo{attachedFiles.length > 1 ? "s" : ""} staged
                    </span>
                    <button
                      type="button"
                      className="copilot-clear-all-attachments-btn"
                      onClick={handleClearAllAttached}
                      title="Clear all staged photos"
                    >
                      Clear all
                    </button>
                  </div>
                  <div className="copilot-attachment-thumbnails-scroll">
                    {attachedFiles.map((att, idx) => (
                      <div
                        key={att.id}
                        className="copilot-thumbnail-card"
                        title={att.file?.name || `Photo #${idx + 1}`}
                      >
                        <img src={att.previewUrl} alt={`Photo ${idx + 1}`} />
                        <span className="copilot-thumbnail-idx">#{idx + 1}</span>
                        <button
                          type="button"
                          className="copilot-thumbnail-remove-btn"
                          onClick={() => handleRemoveAttached(att.id)}
                          title="Remove photo"
                        >
                          <FiX size={10} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Input Area */}
              <footer className="copilot-footer">
                <div className="copilot-input-wrapper">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    style={{ display: "none" }}
                    onChange={handleFileChange}
                  />

                  <div className="copilot-tool-buttons">
                    <button
                      type="button"
                      className="copilot-tool-btn"
                      onClick={() => fileInputRef.current?.click()}
                      title="Upload reference photo for custom item"
                      disabled={isLoading}
                    >
                      <FiImage size={15} />
                    </button>
                    <button
                      type="button"
                      className={`copilot-tool-btn ${cameraStatus === "active" ? "active" : ""}`}
                      onClick={handleLiveSnapshot}
                      title="Capture live webcam frame as custom item"
                      disabled={cameraStatus !== "active" || isLoading}
                    >
                      <FiCamera size={15} />
                    </button>
                  </div>

                  <textarea
                    ref={inputRef}
                    className="copilot-textarea"
                    placeholder={
                      attachedFiles.length > 0
                        ? `e.g. Save as custom item '${attachedFiles[0]?.name || "My Item"}'...`
                        : "Ask anything... (e.g. 'detect person', 'guard scene')"
                    }
                    rows={1}
                    value={inputPrompt}
                    onChange={(e) => setInputPrompt(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                  />

                  <button
                    type="button"
                    className="copilot-send-btn"
                    onClick={() => handleSend()}
                    disabled={isLoading || (!inputPrompt.trim() && attachedFiles.length === 0)}
                    title="Send instruction (Enter)"
                  >
                    <FiCornerDownLeft size={15} />
                  </button>
                </div>
              </footer>
            </>
          )}
        </aside>
      )}

      {Boolean(cropModalState) && (
        <ObjectBoxModal
          isOpen={Boolean(cropModalState)}
          items={cropModalState?.items || []}
          onCancel={() => {
            if (cropModalState) {
              cropModalState.items.forEach((it) => URL.revokeObjectURL(it.previewUrl));
            }
            setCropModalState(null);
          }}
          onConfirm={async (results) => {
            const current = cropModalState;
            setCropModalState(null);
            if (!current || !results || results.length === 0) return;

            const res = results[0];
            try {
              let finalFile = res.file;
              let finalPreviewUrl = res.croppedPreviewUrl;
              let objectUrlToRevoke = null;

              if (res.croppedPreviewUrl && res.croppedPreviewUrl.startsWith("data:")) {
                const blob = await (await fetch(res.croppedPreviewUrl)).blob();
                const baseName = (current.defaultName || "custom_item")
                  .toLowerCase()
                  .replace(/[^a-z0-9_-]/g, "_");
                finalFile = new File([blob], `${baseName}_ref_${Date.now()}.jpg`, {
                  type: "image/jpeg",
                });
              } else {
                finalPreviewUrl = URL.createObjectURL(finalFile);
                objectUrlToRevoke = finalPreviewUrl;
              }

              const newAttachment = {
                id: "att_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7),
                file: finalFile,
                previewUrl: finalPreviewUrl,
                objectUrlToRevoke,
                name: current.defaultName || "Custom Item",
              };

              const newFiles = [...attachedFiles, newAttachment];
              setAttachedFiles(newFiles);

              // Update prompt if empty or if matching default pattern
              const promptTrimmed = inputPrompt.trim();
              const isDefaultPattern =
                !promptTrimmed ||
                /^Save (this|these) (snapshot|photo|\d+ reference photos) as custom item/i.test(
                  promptTrimmed
                );

              if (isDefaultPattern) {
                const targetName =
                  extractItemName(inputPrompt) || current.defaultName || "My Item";
                const count = newFiles.length;
                if (count === 1) {
                  const actionWord = current.isLiveSnapshot ? "snapshot" : "photo";
                  setInputPrompt(`Save this ${actionWord} as custom item '${targetName}'`);
                } else {
                  setInputPrompt(
                    `Save these ${count} reference photos as custom item '${targetName}'`
                  );
                }
              }

              inputRef.current?.focus();
            } catch (err) {
              console.error("Failed to process cropped file:", err);
            } finally {
              current.items.forEach((it) => URL.revokeObjectURL(it.previewUrl));
            }
          }}
        />
      )}
    </>
  );
}
