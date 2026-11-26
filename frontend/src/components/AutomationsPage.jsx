import React, { useState, useEffect, useRef } from "react";
import {
  FiArrowLeft,
  FiCheck,
  FiMail,
  FiMessageSquare,
  FiPlay,
  FiPlus,
  FiRadio,
  FiSearch,
  FiSliders,
  FiSun,
  FiVolume2,
  FiX,
  FiZap,
} from "react-icons/fi";

export const ALL_EVENT_OPTIONS = [
  { value: "camera_blinded", label: "Camera Blinded / Lens Covered" },
  { value: "camera_moved", label: "Camera Moved / Bumped" },
  { value: "unattended_room", label: "Unattended Room (Empty Scene)" },
  { value: "crowd_detected", label: "High Occupancy / Crowd Alert" },
];

export const ALL_ACTION_OPTIONS = [
  { id: "alarm_siren", label: "Play Audio Siren", icon: FiVolume2 },
  { id: "send_sms", label: "Send Urgent SMS", icon: FiMessageSquare },
  { id: "send_email", label: "Send Detailed Email", icon: FiMail },
  { id: "voice_tts", label: "Spoken Voice Warning (TTS)", icon: FiRadio },
  { id: "screen_flash", label: "Visual Screen Strobe", icon: FiSun },
];

export const DEFAULT_RULES = [];

export function AutomationsPage({
  rules = [],
  onSaveRules,
  onBackToMonitor,
  isGuarded = false,
  onTestRuleActions,
}) {
  const [localRules, setLocalRules] = useState(Array.isArray(rules) ? rules : []);
  const [activeSearchRuleId, setActiveSearchRuleId] = useState(null);
  const [searchQueries, setSearchQueries] = useState({});
  const [saveSuccessNotice, setSaveSuccessNotice] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    if (Array.isArray(rules)) {
      const validEventValues = new Set(ALL_EVENT_OPTIONS.map((e) => e.value));
      const validActionIds = new Set(ALL_ACTION_OPTIONS.map((a) => a.id));
      const sanitized = rules
        .filter((r) => validEventValues.has(r.event))
        .map((r) => {
          const actList = Array.isArray(r.actions)
            ? r.actions
            : r.action
            ? [r.action]
            : [];
          const filteredActs = actList.filter((a) => validActionIds.has(a));
          return {
            ...r,
            actions: filteredActs.length > 0 ? filteredActs : ["alarm_siren"],
          };
        });
      setLocalRules(sanitized);
    }
  }, [rules]);

  // Click outside search dropdown
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

  const currentList = Array.isArray(localRules) ? localRules : [];

  const handleAddRule = () => {
    const usedEvents = new Set(currentList.map((r) => r.event));
    const nextAvailable = ALL_EVENT_OPTIONS.find((opt) => !usedEvents.has(opt.value));
    if (!nextAvailable) return;

    const isImmediateEvent = nextAvailable.value === "camera_moved";

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

  const handleResetDefaults = () => {
    setLocalRules([]);
    onSaveRules([]);
  };

  const handleSaveAndApply = () => {
    onSaveRules(localRules);
    setSaveSuccessNotice(true);
    setTimeout(() => setSaveSuccessNotice(false), 3000);
  };

  const allEventsUsed = ALL_EVENT_OPTIONS.every((opt) =>
    currentList.some((r) => r.event === opt.value)
  );

  return (
    <div className="automations-page-container">
      {/* Top Header & Breadcrumb Bar */}
      <div className="automations-page-header">
        <div className="automations-header-top-nav">
          <button
            type="button"
            className="btn btn-secondary automations-back-btn"
            onClick={onBackToMonitor}
            title="Return to real-time webcam feed"
          >
            <FiArrowLeft />
            <span>Back to Live Monitor</span>
          </button>
          <span className="automations-count-pill">
            {currentList.length} rule{currentList.length !== 1 ? "s" : ""} configured
          </span>
        </div>

        <div className="automations-header-main">
          <div className="automations-title-wrap">
            <div className="automations-title-row">
              <span className="automations-title-icon" aria-hidden="true">
                <FiZap />
              </span>
              <h2>Automations</h2>
            </div>
            <p className="automations-subtitle">
              The actions below will automatically trigger based on live detection events and surveillance rules.
            </p>
          </div>

          <div className="automations-header-actions">
            <button
              type="button"
              className="btn btn-secondary automations-reset-btn"
              onClick={handleResetDefaults}
              title="Clear all rules"
            >
              Reset All
            </button>
            <button
              type="button"
              className="btn btn-primary automations-save-btn"
              onClick={handleSaveAndApply}
            >
              <FiCheck />
              <span>Save &amp; Apply</span>
            </button>
          </div>
        </div>
      </div>

      {/* Save Toast Notice */}
      {saveSuccessNotice && (
        <div className="alert-banner alert-success" style={{ margin: "4px 0 0" }}>
          <FiCheck style={{ fontSize: "1.1rem" }} />
          <span>Automations successfully saved and active in the live surveillance engine!</span>
        </div>
      )}

      {/* Overview Stat Badges (Visible on desktop/tablet, hidden on mobile for clean immediate rule view) */}
      <div className="automations-stats-row">
        <div className="automation-stat-card">
          <div className="stat-label">Active Rules</div>
          <div className="stat-value">{currentList.length}</div>
          <div className="stat-hint">
            {currentList.length === 0 ? "No active trigger rules" : "Ready to evaluate on live feed"}
          </div>
        </div>

        <div className="automation-stat-card">
          <div className="stat-label">Guard Status</div>
          <div className="stat-value" style={{ color: isGuarded ? "#34d399" : "#94a3b8" }}>
            {isGuarded ? "Armed" : "Disarmed"}
          </div>
          <div className="stat-hint">
            {isGuarded ? "Scene baseline is armed for missing items" : "Arm scene on Live Monitor"}
          </div>
        </div>

        <div className="automation-stat-card">
          <div className="stat-label">Available Triggers</div>
          <div className="stat-value">{ALL_EVENT_OPTIONS.length}</div>
          <div className="stat-hint">Camera tamper, empty room &amp; crowd alert</div>
        </div>

        <div className="automation-stat-card">
          <div className="stat-label">Supported Actions</div>
          <div className="stat-value">{ALL_ACTION_OPTIONS.length}</div>
          <div className="stat-hint">Siren, SMS, Email, Voice TTS, Strobe</div>
        </div>
      </div>

      {/* Rules Builder Section */}
      <div className="automations-builder-card">
        <div className="automations-builder-header">
          <h3>Trigger action if</h3>
          <p>Define what events to watch for, condition threshold, and which actions to fire.</p>
        </div>

        {currentList.length === 0 ? (
          <div className="automations-empty-state">
            <div className="empty-icon-circle">
              <FiSliders />
            </div>
            <h4>No rules configured</h4>
            <p>
              Automations allow your camera to sound sirens, send SMS alerts, speak warnings, or email snapshots
              automatically when suspicious events happen.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleAddRule}
              style={{ marginTop: "8px" }}
            >
              <FiPlus />
              <span>Add Your First Rule</span>
            </button>
          </div>
        ) : (
          <div className="automations-rules-list">
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
                  className={`automations-rule-card ${
                    isDropdownOpen ? "has-open-dropdown" : ""
                  }`}
                >
                  {/* Top Row: Event Selector | Condition | Threshold | Operations Search | Delete */}
                  <div className="automations-rule-row">
                    {/* 1. Event Trigger */}
                    <div className="rule-field-wrap rule-field-event">
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
                    </div>

                    {/* 2. Operator */}
                    <div className="rule-field-wrap rule-field-operator">
                      <select
                        className="rules-select"
                        value={rule.operator}
                        onChange={(e) => handleUpdateRule(rule.id, "operator", e.target.value)}
                        aria-label="Condition operator"
                      >
                        <option value="persists_for">persists for more than</option>
                        <option value="immediate">happens immediately</option>
                      </select>
                    </div>

                    {/* 3. Duration / Threshold */}
                    <div className="rule-field-wrap rule-field-threshold">
                      {rule.operator === "persists_for" ? (
                        <div className="threshold-input-group">
                          <input
                            type="number"
                            className="rules-input threshold-number-input"
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
                            aria-label="Duration threshold"
                          />
                          <span className="threshold-unit-label">sec</span>
                        </div>
                      ) : (
                        <div className="threshold-instant-badge">Instant</div>
                      )}
                    </div>

                    {/* 4. Searchable Operations Input */}
                    <div
                      className={`rule-field-wrap rule-field-actions ${
                        isDropdownOpen ? "dropdown-open" : ""
                      }`}
                      style={{ position: "relative" }}
                      ref={isDropdownOpen ? dropdownRef : null}
                    >
                      <div
                        className={`rule-search-box ${isDropdownOpen ? "open" : ""}`}
                        onClick={() => {
                          setActiveSearchRuleId(isDropdownOpen ? null : rule.id);
                          if (!isDropdownOpen) {
                            setSearchQueries({ ...searchQueries, [rule.id]: "" });
                          }
                        }}
                      >
                        <div className="rule-search-inner">
                          <FiSearch className="rule-search-icon" />
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
                            className="rule-search-input"
                          />
                        </div>
                        <span className="rule-search-counter">
                          {ruleActions.length}/{ALL_ACTION_OPTIONS.length}
                        </span>
                      </div>

                      {/* Search Results Dropdown */}
                      {isDropdownOpen && (
                        <div className="rule-search-dropdown-menu">
                          {filteredActionOptions.length === 0 ? (
                            <div className="rule-search-dropdown-empty">
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
                                  className={`rule-search-dropdown-item ${
                                    isSelected ? "selected" : ""
                                  }`}
                                >
                                  <div className="item-label-group">
                                    <IconComp className="item-icon" />
                                    <span>{opt.label}</span>
                                  </div>
                                  {isSelected && <FiCheck className="item-check" />}
                                </div>
                              );
                            })
                          )}
                        </div>
                      )}
                    </div>

                    {/* 5. Delete Rule Button */}
                    <div className="rule-field-delete">
                      <button
                        type="button"
                        onClick={() => handleRemoveRule(rule.id)}
                        aria-label="Remove rule"
                        className="rule-delete-btn"
                        title="Delete rule"
                      >
                        <FiX />
                      </button>
                    </div>
                  </div>

                  {/* Sub-row: Removable Tag Pills */}
                  {ruleActions.length > 0 && (
                    <div className="rule-tags-container">
                      <span className="rule-tags-label">Active responses:</span>
                      {ruleActions.map((actId) => {
                        const actDef = ALL_ACTION_OPTIONS.find((a) => a.id === actId);
                        if (!actDef) return null;
                        const IconComp = actDef.icon;
                        return (
                          <span key={actId} className="rule-action-tag-pill">
                            <IconComp className="tag-icon" />
                            <span className="tag-label">{actDef.label}</span>
                            <button
                              type="button"
                              onClick={() => handleRemoveAction(rule.id, actId)}
                              className="tag-remove-btn"
                              title="Remove operation"
                              aria-label={`Remove ${actDef.label}`}
                            >
                              <FiX style={{ fontSize: "0.85rem" }} />
                            </button>
                          </span>
                        );
                      })}

                      {typeof onTestRuleActions === "function" && (
                        <button
                          type="button"
                          onClick={() => onTestRuleActions(rule)}
                          className="rule-test-btn"
                          title="Test trigger these actions right now"
                        >
                          <FiPlay size={11} /> Test Action Responses
                        </button>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* + ADD Button (Centered as in reference images) */}
        <div className="automations-add-wrap">
          <button
            type="button"
            onClick={handleAddRule}
            disabled={allEventsUsed}
            className="automations-add-btn"
            title={
              allEventsUsed
                ? "All available events have already been configured"
                : "Add a new automation rule"
            }
          >
            <FiPlus style={{ fontSize: "1rem" }} />
            <span>
              {allEventsUsed
                ? `All Rules Added (${ALL_EVENT_OPTIONS.length}/${ALL_EVENT_OPTIONS.length})`
                : "ADD"}
            </span>
          </button>
        </div>
      </div>

      {/* Bottom Action Bar */}
      <div className="automations-bottom-bar">
        <button
          type="button"
          className="btn btn-secondary automations-bottom-back"
          onClick={onBackToMonitor}
        >
          <FiArrowLeft /> Back to Live Monitor
        </button>

        <button
          type="button"
          className="btn btn-primary automations-bottom-save"
          onClick={handleSaveAndApply}
        >
          <FiCheck /> Save &amp; Apply Changes
        </button>
      </div>
    </div>
  );
}
