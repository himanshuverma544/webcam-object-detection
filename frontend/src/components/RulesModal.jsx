import React, { useState, useEffect, useRef } from "react";
import {
  FiAlertTriangle,
  FiCamera,
  FiCheck,
  FiFileText,
  FiMail,
  FiMessageSquare,
  FiPlus,
  FiRadio,
  FiSearch,
  FiSend,
  FiSun,
  FiVolume2,
  FiX,
} from "react-icons/fi";

export const ALL_EVENT_OPTIONS = [
  { value: "camera_blinded", label: "Camera Blinded / Lens Covered" },
  { value: "camera_moved", label: "Camera Moved / Bumped" },
  { value: "item_missing", label: "Guarded Item Gone Missing" },
  { value: "item_moved", label: "Guarded Item Displaced / Moved" },
  { value: "item_restored", label: "Guarded Item Restored / Recovered" },
  { value: "person_detected", label: "Person Intrusion Detected" },
  { value: "proximity_alert", label: "Proximity Alert (Person Near Item)" },
  { value: "unattended_room", label: "Unattended Room (Empty Scene)" },
  { value: "crowd_detected", label: "High Occupancy / Crowd Alert" },
  { value: "tripwire_entry", label: "Restricted Zone / Tripwire Entry" },
  { value: "abandoned_object", label: "Unattended / Abandoned Object" },
];

export const ALL_ACTION_OPTIONS = [
  { id: "alarm_siren", label: "Play Audio Siren", icon: FiVolume2 },
  { id: "send_sms", label: "Send Urgent SMS", icon: FiMessageSquare },
  { id: "send_email", label: "Send Detailed Email", icon: FiMail },
  { id: "voice_tts", label: "Spoken Voice Warning (TTS)", icon: FiRadio },
  { id: "screen_flash", label: "Visual Screen Strobe", icon: FiSun },
  { id: "capture_snapshot", label: "Capture Evidence Snapshot", icon: FiCamera },
  { id: "log_incident", label: "Incident Log Entry", icon: FiFileText },
  { id: "webhook_dispatch", label: "Webhook Dispatch", icon: FiSend },
];

export const DEFAULT_RULES = [];

export function RulesModal({ isOpen, onClose, rules, onSaveRules }) {
  const safeInitialRules = Array.isArray(rules) ? rules : DEFAULT_RULES;
  const [localRules, setLocalRules] = useState(safeInitialRules);
  const [activeSearchRuleId, setActiveSearchRuleId] = useState(null);
  const [searchQueries, setSearchQueries] = useState({});
  const dropdownRef = useRef(null);

  useEffect(() => {
    if (Array.isArray(rules)) {
      setLocalRules(rules);
    }
  }, [rules, isOpen]);

  // Close search dropdown on click outside
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setActiveSearchRuleId(null);
        setSearchQueries({});
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const currentList = Array.isArray(localRules) ? localRules : DEFAULT_RULES;

  const handleAddRule = () => {
    const usedEvents = new Set(currentList.map((r) => r.event));
    const nextAvailable = ALL_EVENT_OPTIONS.find((opt) => !usedEvents.has(opt.value));
    if (!nextAvailable) return;

    const isImmediateEvent =
      nextAvailable.value === "camera_moved" ||
      nextAvailable.value === "item_moved" ||
      nextAvailable.value === "item_restored" ||
      nextAvailable.value === "person_detected" ||
      nextAvailable.value === "proximity_alert" ||
      nextAvailable.value === "tripwire_entry";

    const newRule = {
      id: `rule_${Date.now()}`,
      event: nextAvailable.value,
      operator: isImmediateEvent ? "immediate" : "persists_for",
      threshold: isImmediateEvent ? 1 : 3,
      unit: "sec",
      actions: ["alarm_siren"],
      enabled: true,
    };
    setLocalRules([...currentList, newRule]);
  };

  const handleUpdateRule = (id, field, value) => {
    setLocalRules((prev) =>
      prev.map((r) => (r.id === id ? { ...r, [field]: value } : r))
    );
  };

  const handleToggleAction = (ruleId, actionId) => {
    setLocalRules((prev) =>
      prev.map((r) => {
        if (r.id !== ruleId) return r;
        const currentActions = Array.isArray(r.actions)
          ? r.actions
          : r.action
          ? [r.action]
          : [];
        const nextActions = currentActions.includes(actionId)
          ? currentActions.filter((a) => a !== actionId)
          : [...currentActions, actionId];
        return { ...r, actions: nextActions };
      })
    );
  };

  const handleRemoveAction = (ruleId, actionId) => {
    setLocalRules((prev) =>
      prev.map((r) => {
        if (r.id !== ruleId) return r;
        const currentActions = Array.isArray(r.actions) ? r.actions : [];
        return { ...r, actions: currentActions.filter((a) => a !== actionId) };
      })
    );
  };

  const handleRemoveRule = (id) => {
    setLocalRules((prev) => prev.filter((r) => r.id !== id));
  };

  const handleReset = () => {
    setLocalRules(DEFAULT_RULES);
  };

  const handleSave = () => {
    onSaveRules(localRules);
    onClose();
  };

  const allEventsUsed = ALL_EVENT_OPTIONS.every((opt) =>
    currentList.some((r) => r.event === opt.value)
  );

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" style={{ zIndex: 1100 }}>
      <div
        className="modal-card rules-modal-card"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: "1180px", width: "95%" }}
      >
        {/* Header */}
        <div className="modal-header">
          <div className="modal-header-info">
            <div className="modal-title-row">
              <h2>Rule</h2>
            </div>
            <p className="modal-subtitle" style={{ fontSize: "0.9rem", color: "var(--text-muted, #94a3b8)", marginTop: "4px" }}>
              The actions will be triggered based on the rules in this section.
            </p>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close rules modal"
          >
            <FiX style={{ fontSize: "1.2rem" }} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="rules-modal-body" style={{ padding: "24px", maxHeight: "68vh", overflowY: "auto" }}>
          <div style={{ marginBottom: "16px", fontWeight: "600", fontSize: "0.95rem", color: "var(--text-primary, #f1f5f9)" }}>
            Trigger action if
          </div>

          {currentList.length === 0 ? (
            <div style={{ textAlign: "center", padding: "32px 16px", color: "var(--text-muted, #94a3b8)" }}>
              No rules configured. Click <strong>+ ADD</strong> below to create your first rule.
            </div>
          ) : (
            <div className="rules-list-container" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
              {currentList.map((rule) => {
                const ruleActions = Array.isArray(rule.actions)
                  ? rule.actions
                  : rule.action
                  ? [rule.action]
                  : [];
                const currentSearch = searchQueries[rule.id] || "";
                const isDropdownOpen = activeSearchRuleId === rule.id;

                const filteredActionOptions = ALL_ACTION_OPTIONS.filter((opt) =>
                  opt.label.toLowerCase().includes(currentSearch.toLowerCase())
                );

                return (
                  <div
                    key={rule.id}
                    className="rules-builder-card"
                    style={{
                      background: "rgba(255, 255, 255, 0.03)",
                      border: "1px solid var(--border-subtle, rgba(255, 255, 255, 0.1))",
                      borderRadius: "8px",
                      padding: "12px 14px",
                      display: "flex",
                      flexDirection: "column",
                      gap: "8px",
                    }}
                  >
                    {/* Top Row: Event | Condition | Threshold | Search Operations | Remove */}
                    <div
                      className="rules-builder-row"
                      style={{
                        display: "grid",
                        gridTemplateColumns: "minmax(270px, 1.5fr) minmax(210px, 1.2fr) minmax(110px, 0.7fr) minmax(240px, 1.4fr) auto",
                        gap: "12px",
                        alignItems: "center",
                      }}
                    >
                      {/* 1. Event Trigger (Disabled if selected elsewhere, without extra text) */}
                      <select
                        className="rules-select"
                        value={rule.event}
                        onChange={(e) => handleUpdateRule(rule.id, "event", e.target.value)}
                        aria-label="Event trigger"
                      >
                        {ALL_EVENT_OPTIONS.map((opt) => {
                          const isSelectedElsewhere = currentList.some(
                            (other) => other.id !== rule.id && other.event === opt.value
                          );
                          return (
                            <option
                              key={opt.value}
                              value={opt.value}
                              disabled={isSelectedElsewhere}
                              style={{
                                color: isSelectedElsewhere ? "#64748b" : "inherit",
                              }}
                            >
                              {opt.label}
                            </option>
                          );
                        })}
                      </select>

                      {/* 2. Operator */}
                      <select
                        className="rules-select"
                        value={rule.operator}
                        onChange={(e) => handleUpdateRule(rule.id, "operator", e.target.value)}
                        aria-label="Condition operator"
                      >
                        <option value="persists_for">persists for more than</option>
                        <option value="immediate">happens immediately</option>
                      </select>

                      {/* 3. Threshold / Duration */}
                      {rule.operator === "persists_for" ? (
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <input
                            type="number"
                            className="rules-input"
                            min="1"
                            max="300"
                            value={rule.threshold}
                            onChange={(e) =>
                              handleUpdateRule(
                                rule.id,
                                "threshold",
                                Math.max(1, parseInt(e.target.value) || 1)
                              )
                            }
                            style={{ width: "100%", textAlign: "center" }}
                            aria-label="Duration threshold"
                          />
                          <span style={{ fontSize: "0.82rem", color: "var(--text-muted, #94a3b8)", whiteSpace: "nowrap" }}>
                            sec
                          </span>
                        </div>
                      ) : (
                        <div
                          style={{
                            padding: "6px 12px",
                            background: "rgba(255, 255, 255, 0.05)",
                            borderRadius: "6px",
                            fontSize: "0.82rem",
                            color: "var(--text-muted, #94a3b8)",
                            textAlign: "center",
                          }}
                        >
                          Instant
                        </div>
                      )}

                      {/* 4. Searchable Operations Input */}
                      <div
                        style={{ position: "relative" }}
                        ref={isDropdownOpen ? dropdownRef : null}
                      >
                        <div
                          onClick={() => {
                            setActiveSearchRuleId(isDropdownOpen ? null : rule.id);
                          }}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            background: "#182234",
                            border: isDropdownOpen
                              ? "1px solid #6366f1"
                              : "1px solid rgba(255, 255, 255, 0.18)",
                            borderRadius: "6px",
                            padding: "7px 10px",
                            cursor: "pointer",
                            fontSize: "0.85rem",
                            color: "#f1f5f9",
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: "6px", width: "100%" }}>
                            <FiSearch style={{ color: "var(--text-muted, #94a3b8)", flexShrink: 0 }} />
                            <input
                              type="text"
                              placeholder="Search Operations..."
                              value={isDropdownOpen ? currentSearch : ""}
                              onChange={(e) => {
                                setSearchQueries({ ...searchQueries, [rule.id]: e.target.value });
                                setActiveSearchRuleId(rule.id);
                              }}
                              onFocus={() => setActiveSearchRuleId(rule.id)}
                              onClick={(e) => e.stopPropagation()}
                              style={{
                                background: "transparent",
                                border: "none",
                                color: "#f1f5f9",
                                outline: "none",
                                width: "100%",
                                fontSize: "0.85rem",
                              }}
                            />
                          </div>
                          <span
                            style={{
                              fontSize: "0.76rem",
                              color: "var(--text-muted, #94a3b8)",
                              whiteSpace: "nowrap",
                              marginLeft: "6px",
                            }}
                          >
                            {ruleActions.length}/{ALL_ACTION_OPTIONS.length}
                          </span>
                        </div>

                        {/* Search Results Dropdown */}
                        {isDropdownOpen && (
                          <div
                            style={{
                              position: "absolute",
                              top: "calc(100% + 4px)",
                              left: 0,
                              right: 0,
                              background: "#1e293b",
                              border: "1px solid rgba(255, 255, 255, 0.2)",
                              borderRadius: "6px",
                              boxShadow: "0 10px 25px rgba(0,0,0,0.5)",
                              zIndex: 1200,
                              maxHeight: "180px",
                              overflowY: "auto",
                            }}
                          >
                            {filteredActionOptions.length === 0 ? (
                              <div style={{ padding: "8px 12px", fontSize: "0.8rem", color: "#94a3b8" }}>
                                No matching operations
                              </div>
                            ) : (
                              filteredActionOptions.map((opt) => {
                                const isSelected = ruleActions.includes(opt.id);
                                const IconComp = opt.icon;
                                return (
                                  <div
                                    key={opt.id}
                                    onClick={() => handleToggleAction(rule.id, opt.id)}
                                    style={{
                                      display: "flex",
                                      alignItems: "center",
                                      justifyContent: "space-between",
                                      padding: "8px 12px",
                                      cursor: "pointer",
                                      fontSize: "0.84rem",
                                      color: isSelected ? "#38bdf8" : "#f1f5f9",
                                      background: isSelected ? "rgba(56, 189, 248, 0.12)" : "transparent",
                                      borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
                                    }}
                                  >
                                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                      <IconComp style={{ fontSize: "1rem" }} />
                                      <span>{opt.label}</span>
                                    </div>
                                    {isSelected && <FiCheck style={{ color: "#38bdf8" }} />}
                                  </div>
                                );
                              })
                            )}
                          </div>
                        )}
                      </div>

                      {/* 5. Remove Button */}
                      <button
                        type="button"
                        onClick={() => handleRemoveRule(rule.id)}
                        aria-label="Remove rule"
                        style={{
                          background: "none",
                          border: "none",
                          color: "var(--text-muted, #94a3b8)",
                          cursor: "pointer",
                          fontSize: "1.1rem",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                        title="Remove rule"
                      >
                        <FiX />
                      </button>
                    </div>

                    {/* Sub-row: Selected Action Pills (as in reference image) */}
                    {ruleActions.length > 0 && (
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: "8px",
                          paddingTop: "2px",
                        }}
                      >
                        {ruleActions.map((actId) => {
                          const actDef = ALL_ACTION_OPTIONS.find((a) => a.id === actId);
                          if (!actDef) return null;
                          const IconComp = actDef.icon;
                          return (
                            <span
                              key={actId}
                              style={{
                                display: "inline-flex",
                                alignItems: "center",
                                gap: "6px",
                                background: "rgba(255, 255, 255, 0.07)",
                                border: "1px solid rgba(255, 255, 255, 0.14)",
                                borderRadius: "6px",
                                padding: "4px 8px 4px 10px",
                                fontSize: "0.82rem",
                                color: "#e2e8f0",
                              }}
                            >
                              <IconComp style={{ fontSize: "0.9rem", color: "#38bdf8" }} />
                              <span>{actDef.label}</span>
                              <button
                                type="button"
                                onClick={() => handleRemoveAction(rule.id, actId)}
                                style={{
                                  background: "none",
                                  border: "none",
                                  color: "#94a3b8",
                                  cursor: "pointer",
                                  display: "inline-flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  padding: "0 2px",
                                  lineHeight: 1,
                                }}
                                title="Remove operation"
                              >
                                <FiX style={{ fontSize: "0.8rem" }} />
                              </button>
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* + ADD Button */}
          <div style={{ marginTop: "16px" }}>
            <button
              type="button"
              onClick={handleAddRule}
              disabled={allEventsUsed}
              style={{
                background: allEventsUsed ? "rgba(255, 255, 255, 0.02)" : "rgba(255, 255, 255, 0.06)",
                border: "1px dashed var(--border-subtle, rgba(255, 255, 255, 0.2))",
                borderRadius: "8px",
                padding: "8px 24px",
                color: allEventsUsed ? "var(--text-muted, #64748b)" : "var(--text-primary, #f1f5f9)",
                cursor: allEventsUsed ? "not-allowed" : "pointer",
                fontWeight: "600",
                fontSize: "0.88rem",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                transition: "all 0.15s ease",
                opacity: allEventsUsed ? 0.6 : 1,
              }}
              title={
                allEventsUsed
                  ? "All available events have already been configured"
                  : "Add a new event rule"
              }
            >
              <FiPlus style={{ fontSize: "0.95rem" }} />
              <span>
                {allEventsUsed
                  ? `All Event Rules Added (${ALL_EVENT_OPTIONS.length}/${ALL_EVENT_OPTIONS.length})`
                  : "ADD"}
              </span>
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="modal-footer" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleReset}
            title="Reset rules to default baseline"
          >
            Reset Defaults
          </button>

          <div style={{ display: "flex", gap: "10px" }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSave}
            >
              Save &amp; Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
