import React from "react";

export function MissingAlertBanner({ missingItems = [], onDismiss }) {
  if (!Array.isArray(missingItems) || missingItems.length === 0) {
    return null;
  }

  const hasGuardedBreach = missingItems.some((item) => item.isGuarded);

  return (
    <div
      className={`missing-alert-container ${hasGuardedBreach ? "breach-alarm" : "warning-notice"}`}
      role="alert"
      aria-live="assertive"
    >
      <div className="missing-alert-header">
        <div className="alert-title-wrap">
          <span className="alert-siren-icon">{hasGuardedBreach ? "🚨" : "⚠️"}</span>
          <div>
            <h4 className="alert-heading">
              {hasGuardedBreach
                ? `GUARD BREACH: ${missingItems.length} item(s) missing from armed scene`
                : `${missingItems.length} Missing Item(s) Detected`}
            </h4>
            <p className="alert-desc">
              Items removed from their positions are outlined with red footprints on the video feed.
            </p>
          </div>
        </div>
      </div>

      <div className="missing-items-chips">
        {missingItems.map((item) => (
          <div key={item.id} className="missing-item-chip">
            <div className="missing-chip-info">
              {item.isGuarded && <span className="guarded-mini-badge">🛡️ Armed</span>}
              <strong className="missing-chip-name">{item.label}</strong>
              <span className="missing-chip-time">
                Missing since{" "}
                {item.missingSince
                  ? new Date(item.missingSince).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })
                  : "recently"}
              </span>
            </div>
            <button
              type="button"
              className="btn-dismiss-missing"
              onClick={() => onDismiss(item.id)}
              title="Acknowledge and dismiss alert"
            >
              Dismiss ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

