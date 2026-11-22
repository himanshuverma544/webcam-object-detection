import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  FiCamera,
  FiCheckCircle,
  FiCornerDownLeft,
  FiImage,
  FiMaximize2,
  FiMessageSquare,
  FiMinimize2,
  FiTrash2,
  FiX,
  FiZap,
} from "react-icons/fi";
import { sendChatMessage } from "../services/api";

const QUICK_SUGGESTIONS = [
  { label: "📹 Start Camera", prompt: "Start the camera" },
  { label: "👤 Person & Phone", prompt: "Only detect person and cell phone" },
  { label: "🛡️ Guard Scene", prompt: "Guard the scene now" },
  { label: "🎯 70% Confidence", prompt: "Set confidence to 70%" },
  { label: "📸 Snapshot Item", prompt: "Capture snapshot and save as 'My Mug'" },
  { label: "📊 Show Metrics", prompt: "Show session metrics summary" },
];

export function ChatCopilot({
  cameraStatus,
  isDetecting,
  confidence,
  selectedClasses,
  customItems,
  selectedCustomIds,
  isGuarded,
  supportedClasses,
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
  const [attachedFile, setAttachedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
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

  // Clean up object URL when attachedFile changes
  useEffect(() => {
    if (attachedFile) {
      const url = URL.createObjectURL(attachedFile);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl(null);
    }
  }, [attachedFile]);

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
    if (file) {
      setAttachedFile(file);
      if (!inputPrompt.trim()) {
        const cleanName = file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ");
        setInputPrompt(`Save this photo as custom item '${cleanName}'`);
      }
    }
    e.target.value = "";
  };

  const handleLiveSnapshot = async () => {
    if (!onCaptureLiveSnapshot || cameraStatus !== "active") {
      alert("Please start the webcam first before taking a live snapshot!");
      return;
    }
    try {
      const file = await onCaptureLiveSnapshot("live_snapshot");
      if (file) {
        setAttachedFile(file);
        if (!inputPrompt.trim()) {
          setInputPrompt("Save this snapshot as custom item 'My Item'");
        }
      }
    } catch (err) {
      console.error("Failed to capture snapshot:", err);
    }
  };

  const handleSend = async (overridePrompt) => {
    const textToSend = (overridePrompt !== undefined ? overridePrompt : inputPrompt).trim();
    const fileToSend = attachedFile;

    if (!textToSend && !fileToSend) return;

    const userMessageId = "user_" + Date.now();
    const userMsg = {
      id: userMessageId,
      role: "user",
      content: textToSend || (fileToSend ? `[Uploaded image: ${fileToSend.name}]` : ""),
      previewUrl: previewUrl,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputPrompt("");
    setAttachedFile(null);
    setIsLoading(true);

    // Prepare current UI state context for copilot
    const currentState = {
      cameraStatus,
      isDetecting,
      confidence,
      selectedClasses,
      customItemCount: customItems.length,
      selectedCustomIds,
      isGuarded,
      supportedClassesCount: supportedClasses?.length || 80,
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
        file: fileToSend,
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

      // Execute actions in the parent React application
      if (Array.isArray(response.actions) && response.actions.length > 0 && onExecuteActions) {
        onExecuteActions(response.actions);
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
          content: `⚠️ **Error communicating with AI Copilot:** ${err.message || "Request failed"}. Please try again.`,
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
                      {m.previewUrl && (
                        <div className="copilot-msg-attachment">
                          <img src={m.previewUrl} alt="Attached input" />
                        </div>
                      )}

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

              {/* Attachment Preview Box */}
              {previewUrl && (
                <div className="copilot-attachment-bar">
                  <div className="copilot-attachment-preview">
                    <img src={previewUrl} alt="Ready to upload" />
                    <span className="copilot-attachment-name">
                      {attachedFile?.name || "Snapshot frame"}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="copilot-remove-attachment-btn"
                    onClick={() => setAttachedFile(null)}
                    title="Remove attachment"
                  >
                    <FiX size={14} />
                  </button>
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
                      attachedFile
                        ? "e.g. Save as custom item 'My Backpack'..."
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
                    disabled={isLoading || (!inputPrompt.trim() && !attachedFile)}
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
    </>
  );
}

