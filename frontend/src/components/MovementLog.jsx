import React, { useMemo, useState } from "react";
import { getClassColor } from "./VideoViewport";

export function MovementLog({
  events = [],
  missingEvents = [],
  activeMovingCount = 0,
  missingItemsCount = 0,
  onClearEvents,
  minMoveDistance,
  settleDwellTimeMs,
}) {
  const [filterTab, setFilterTab] = useState("all"); // 'all' | 'moved' | 'missing'

  // Combine and sort all events by timestamp descending
  const combinedEvents = useMemo(() => {
    const list = [
      ...events.map((e) => ({ ...e, eventCategory: "MOVED" })),
      ...missingEvents.map((e) => ({ ...e, eventCategory: e.type || "MISSING" })),
    ];
    list.sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return tb - ta;
    });
    return list;
  }, [events, missingEvents]);

  const filteredEvents = useMemo(() => {
    if (filterTab === "moved") {
      return combinedEvents.filter((e) => e.eventCategory === "MOVED");
    }
    if (filterTab === "missing") {
      return combinedEvents.filter((e) => e.eventCategory === "MISSING" || e.eventCategory === "RESTORED");
    }
    return combinedEvents;
  }, [combinedEvents, filterTab]);

  return (
    <div className="movement-log-card" aria-label="Object activity and security history">
      <div className="movement-log-header">
        <div className="movement-log-title-group">
          <div className="title-with-pulse">
            <h3>Activity & Security Log</h3>
            {activeMovingCount > 0 && (
              <span className="live-motion-badge" title="Objects currently in motion">
                <span className="pulse-dot" />
                {activeMovingCount} moving
              </span>
            )}
            {missingItemsCount > 0 && (
              <span className="live-missing-badge" title="Objects currently missing">
                <span className="pulse-dot red-pulse" />
                {missingItemsCount} missing
              </span>
            )}
          </div>
          <p className="movement-log-subtitle">
            Relocations (&gt;{minMoveDistance}px) &amp; missing item security events
          </p>
        </div>

        {combinedEvents.length > 0 && (
          <button
            type="button"
            className="btn-clear-events"
            onClick={onClearEvents}
            title="Clear activity log"
          >
            Clear ({combinedEvents.length})
          </button>
        )}
      </div>

      <div className="activity-tabs-row">
        <button
          type="button"
          className={`tab-btn ${filterTab === "all" ? "tab-active" : ""}`}
          onClick={() => setFilterTab("all")}
        >
          All ({combinedEvents.length})
        </button>
        <button
          type="button"
          className={`tab-btn ${filterTab === "moved" ? "tab-active" : ""}`}
          onClick={() => setFilterTab("moved")}
        >
          Relocations ({events.length})
        </button>
        <button
          type="button"
          className={`tab-btn ${filterTab === "missing" ? "tab-active" : ""}`}
          onClick={() => setFilterTab("missing")}
        >
          Security ({missingEvents.length})
        </button>
      </div>

      <div className="movement-events-scroll">
        {filteredEvents.length === 0 ? (
          <div className="empty-movement-state">
            <div className="empty-movement-icon" aria-hidden="true">
              ⇄
            </div>
            <p>No activity detected yet.</p>
            <span className="empty-hint">
              Move items across the screen or remove an item to view real-time relocation
              and missing alerts.
            </span>
          </div>
        ) : (
          <ul className="movement-events-list">
            {filteredEvents.map((event) => {
              const color = getClassColor(event.class_id);
              const timeString = event.timestamp
                ? new Date(event.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })
                : "Just now";

              if (event.eventCategory === "MISSING") {
                const posX = Math.round(event.position?.x || 0);
                const posY = Math.round(event.position?.y || 0);

                return (
                  <li key={event.id} className="movement-event-item item-missing-event">
                    <div className="movement-event-top">
                      <div className="movement-item-badge">
                        <span className="event-type-pill missing-pill">
                          {event.isGuarded ? "🛡️ GUARD BREACH" : "⚠️ MISSING"}
                        </span>
                        <strong className="movement-label">{event.label}</strong>
                      </div>
                      <time className="movement-timestamp">{timeString}</time>
                    </div>

                    <div className="movement-metrics-row">
                      <span className="missing-alert-text">
                        Item removed from spot ({posX}, {posY})
                      </span>
                    </div>
                  </li>
                );
              }

              if (event.eventCategory === "RESTORED") {
                const posX = Math.round(event.position?.x || 0);
                const posY = Math.round(event.position?.y || 0);

                return (
                  <li key={event.id} className="movement-event-item item-restored-event">
                    <div className="movement-event-top">
                      <div className="movement-item-badge">
                        <span className="event-type-pill restored-pill">
                          ✅ RESTORED
                        </span>
                        <strong className="movement-label">{event.label}</strong>
                      </div>
                      <time className="movement-timestamp">{timeString}</time>
                    </div>

                    <div className="movement-metrics-row">
                      <span className="restored-alert-text">
                        Item returned to spot ({posX}, {posY})
                      </span>
                    </div>
                  </li>
                );
              }

              // Standard Relocation MOVED event
              const fromX = Math.round(event.from?.x || 0);
              const fromY = Math.round(event.from?.y || 0);
              const toX = Math.round(event.to?.x || 0);
              const toY = Math.round(event.to?.y || 0);

              return (
                <li key={event.id} className="movement-event-item">
                  <div className="movement-event-top">
                    <div className="movement-item-badge">
                      <span
                        className="movement-color-indicator"
                        style={{ backgroundColor: color }}
                      />
                      <strong className="movement-label">{event.label}</strong>
                      {event.trackId && (
                        <span className="movement-track-id">
                          #{event.trackId.toString().replace(/^bt-|^temp-/, "")}
                        </span>
                      )}
                    </div>
                    <time className="movement-timestamp">{timeString}</time>
                  </div>

                  <div className="movement-metrics-row">
                    <span className="displacement-tag">
                      <span className="metric-arrow">➔</span>
                      <strong>{event.distance} px</strong> moved
                    </span>
                    <span className="duration-tag">{event.durationSec}s travel</span>
                  </div>

                  <div className="movement-coords-row">
                    <span className="coord-point origin-point" title="Original position">
                      From ({fromX}, {fromY})
                    </span>
                    <span className="coord-sep">➔</span>
                    <span className="coord-point dest-point" title="New settled position">
                      To ({toX}, {toY})
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
