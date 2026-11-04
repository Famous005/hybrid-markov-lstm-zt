"""
generate_dataset.py
-------------------
Generates a synthetic Zero Trust policy violation dataset of 6,016 events
matching the thesis specification:
  - 3 classes: 0=Normal, 1=Suspicious, 2=Malicious
  - Class distribution: 96.4% / 2.5% / 1.1%
  - 57 raw features across 6 categories
  - Realistic temporal patterns (attacks peak 1AM-3AM)
  - 5-stage attack progression for malicious sequences
  - All policy violation artifacts the thesis references

Zero Trust concept:
  In a Zero Trust Architecture (NIST SP 800-207), every access request
  is logged by the Policy Enforcement Point (PEP). Denied requests and
  authentication failures become 'policy violation artifacts' - the raw
  material this framework uses for prediction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Fixed random seed for reproducibility (thesis requirement)
SEED = 42
np.random.seed(SEED)

# ── Dataset composition (matches Table 3.1 and Table 4.1 in thesis) ──
TOTAL_EVENTS   = 6016
N_NORMAL       = 5800   # 96.4%
N_SUSPICIOUS   = 150    # 2.5%
N_MALICIOUS    = 66     # 1.1%

# ── Environment parameters ──
N_COMPUTERS    = 28     # distinct machines in simulated ZT environment
N_USERS        = 45
N_SESSIONS     = 320

COMPUTERS = [f"WS-{i:03d}" if i <= 20 else f"SRV-{i:03d}"
             for i in range(1, N_COMPUTERS + 1)]

PRIVILEGE_LEVELS = ["standard", "power_user", "administrator", "domain_admin"]

EVENT_IDS = {
    "EventID_1.0": "Process creation",
    "EventID_3.0": "Network connection",
    "EventID_5.0": "Authentication failure",
    "EventID_7.0": "Denied access",
    "EventID_9.0": "Successful login",
    "EventID_11.0": "Privilege escalation",
    "EventID_13.0": "File access",
    "EventID_15.0": "Registry modification",
    "EventID_17.0": "Scheduled task",
    "EventID_19.0": "Service installation",
}

VIOLATION_TYPES = [
    "failed_authentication",
    "denied_access",
    "privilege_escalation_attempt",
    "policy_bypass_attempt",
    "unauthorized_lateral_access",
    "credential_stuffing",
    "brute_force",
    "none",
]

DESTINATION_PORT_NAMES = [
    "SMB", "RDP", "SSH", "HTTP", "HTTPS",
    "LDAP", "KERBEROS", "WMI", "DNS", "FTP"
]


# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def random_ip(private=True):
    """Generate a random IP address."""
    if private:
        return f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}"
    return f"{np.random.randint(1,223)}.{np.random.randint(0,255)}." \
           f"{np.random.randint(0,255)}.{np.random.randint(1,254)}"


def business_hour_time(base_date):
    """Return a timestamp during business hours (9AM-5PM)."""
    hour   = np.random.randint(9, 17)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return base_date.replace(hour=hour, minute=minute, second=second)


def off_hour_time(base_date, attack_peak=True):
    """
    Return a timestamp outside business hours.
    attack_peak=True biases toward 1AM-3AM (thesis finding: 23.4% of
    malicious events occur at 2AM).
    """
    if attack_peak and np.random.random() < 0.60:
        hour = np.random.choice([1, 2, 3], p=[0.30, 0.50, 0.20])
    else:
        hour = np.random.choice(
            list(range(0, 9)) + list(range(17, 24))
        )
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return base_date.replace(hour=hour, minute=minute, second=second)


def build_base_row(timestamp, label, session_id, user_id,
                   computer, event_id, violation_type,
                   privilege_level, source_ip, dest_ip,
                   port_name, process_id):
    """Build one event row with all 57 columns."""

    hour   = timestamp.hour
    minute = timestamp.minute

    # ── Event ID one-hot columns (20 columns: EventID_1.0 … EventID_19.0) ──
    event_id_cols = {}
    for eid in EVENT_IDS:
        event_id_cols[eid] = 1.0 if eid == event_id else 0.0

    # ── Computer one-hot columns (8 indicator columns, grouped) ──
    comp_index = COMPUTERS.index(computer) if computer in COMPUTERS else 0
    comp_group = comp_index // 4          # 7 groups of 4
    computer_cols = {f"Computer_{g}": 1 if g == comp_group else 0
                     for g in range(7)}
    computer_cols["Computer_other"] = 0

    # ── Network columns (18 columns) ──
    network_cols = {}
    for pname in DESTINATION_PORT_NAMES:
        network_cols[f"DestinationPortName_{pname}"] = (
            1 if pname == port_name else 0
        )
    network_cols["SourceIsIpv6_false"] = 1
    network_cols["SourceIsIpv6_true"]  = 0
    network_cols["Initiated_true"]     = int(label > 0)
    network_cols["Initiated_false"]    = int(label == 0)
    network_cols["DestinationIsIpv6_false"] = 1
    network_cols["DestinationIsIpv6_true"]  = 0
    network_cols["ConnectionType_direct"]   = int(label == 0)
    network_cols["ConnectionType_lateral"]  = int(label == 2)

    # ── Process columns (3 columns) ──
    process_cols = {
        "ProcessId":             process_id,
        "Execution_ProcessID":   process_id + np.random.randint(0, 5),
        "ParentProcessId":       process_id - np.random.randint(1, 3),
    }

    # ── Severity based on label ──
    if label == 0:
        severity = np.random.choice(["low", "info"],    p=[0.3, 0.7])
    elif label == 1:
        severity = np.random.choice(["medium", "high"], p=[0.7, 0.3])
    else:
        severity = np.random.choice(["high", "critical"], p=[0.4, 0.6])

    # ── Action taken ──
    if label == 0:
        action = np.random.choice(["allowed", "logged"], p=[0.9, 0.1])
    elif label == 1:
        action = np.random.choice(["blocked", "alerted", "logged"],
                                  p=[0.5, 0.3, 0.2])
    else:
        action = np.random.choice(["blocked", "alerted"], p=[0.6, 0.4])

    row = {
        # ── Temporal (4 columns) ──
        "SystemTime":        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "SystemTime_hour":   hour,
        "SystemTime_minute": minute,
        "EventRecordID":     np.random.randint(100000, 999999),  # index col

        # ── Unnamed index cols (4 columns, dropped in preprocessing) ──
        "Unnamed: 0":   np.random.randint(0, 6016),
        "Unnamed: 0.1": np.random.randint(0, 6016),
        "Unnamed: 0.2": np.random.randint(0, 6016),
        "Unnamed: 0.3": np.random.randint(0, 6016),

        # ── Identifiers ──
        "session_id":       session_id,
        "user_id":          user_id,
        "source_ip":        source_ip,
        "destination_ip":   dest_ip,
        "Computer":         computer,
        "privilege_level":  privilege_level,
        "violation_type":   violation_type,
        "severity":         severity,
        "action_taken":     action,

        # ── Target label ──
        "Label": label,
    }

    # Merge all column groups
    row.update(event_id_cols)
    row.update(computer_cols)
    row.update(network_cols)
    row.update(process_cols)

    return row


# ════════════════════════════════════════════════════════════
# EVENT GENERATORS PER CLASS
# ════════════════════════════════════════════════════════════

def generate_normal_event(base_date, session_id, user_id):
    """
    Normal event: legitimate user, business hours, standard privilege,
    no violation. Represents the 96.4% majority class.
    """
    timestamp      = business_hour_time(base_date)
    computer       = np.random.choice(COMPUTERS[:20])   # workstations only
    event_id       = np.random.choice(
        list(EVENT_IDS.keys()),
        p=[0.30, 0.20, 0.05, 0.02, 0.15, 0.02,
           0.10, 0.06, 0.05, 0.05]
    )
    violation_type = "none"
    privilege      = np.random.choice(
        PRIVILEGE_LEVELS, p=[0.70, 0.20, 0.08, 0.02]
    )
    source_ip = random_ip()
    dest_ip   = random_ip()
    pid       = np.random.randint(1000, 9000)

    return build_base_row(
        timestamp, 0, session_id, user_id,
        computer, event_id, violation_type,
        privilege, source_ip, dest_ip,
        np.random.choice(DESTINATION_PORT_NAMES), pid
    )


def generate_suspicious_event(base_date, session_id, user_id):
    """
    Suspicious event: off-hours access, failed authentications,
    some denied access attempts. Represents early reconnaissance.
    Zero Trust concept: Policy Enforcement Point logs every denial —
    these are the 'policy violation artifacts' we predict from.
    """
    timestamp = off_hour_time(base_date, attack_peak=False)
    computer  = np.random.choice(COMPUTERS)
    event_id  = np.random.choice(
        list(EVENT_IDS.keys()),
        p=[0.20, 0.25, 0.20, 0.15, 0.03, 0.03,
           0.05, 0.03, 0.03, 0.03]
    )
    violation_type = np.random.choice(
        ["failed_authentication", "denied_access",
         "credential_stuffing", "brute_force"],
        p=[0.40, 0.30, 0.20, 0.10]
    )
    privilege = np.random.choice(
        PRIVILEGE_LEVELS, p=[0.40, 0.35, 0.20, 0.05]
    )
    source_ip = random_ip()
    dest_ip   = random_ip()
    pid       = np.random.randint(1000, 9000)

    return build_base_row(
        timestamp, 1, session_id, user_id,
        computer, event_id, violation_type,
        privilege, source_ip, dest_ip,
        np.random.choice(DESTINATION_PORT_NAMES), pid
    )


def generate_malicious_sequence(base_date, session_id, user_id):
    """
    Generates a 5-stage attack sequence (thesis Section 4.2.4):
      Stage 1 - Reconnaissance: 3-5 failed authentication attempts
      Stage 2 - Privilege probing: 2-3 denied access to high-value targets
      Stage 3 - Credential exploitation: successful authentication
      Stage 4 - Lateral movement: access to additional systems
      Stage 5 - Target access: final resource compromise

    All events labelled Malicious (2).
    Zero Trust concept: Each stage generates policy violation artifacts
    that the PEP logs independently of endpoint health reporting.
    """
    events    = []
    timestamp = off_hour_time(base_date, attack_peak=True)
    source_ip = random_ip()

    # Stage 1: Reconnaissance (3-5 failed auth attempts)
    n_recon = np.random.randint(3, 6)
    for _ in range(n_recon):
        timestamp += timedelta(seconds=np.random.randint(30, 180))
        events.append(build_base_row(
            timestamp, 2, session_id, user_id,
            np.random.choice(COMPUTERS[:20]),
            "EventID_5.0", "failed_authentication",
            "standard", source_ip, random_ip(),
            "LDAP", np.random.randint(1000, 9000)
        ))

    # Stage 2: Privilege probing (2-3 denied access attempts)
    n_probe = np.random.randint(2, 4)
    for _ in range(n_probe):
        timestamp += timedelta(seconds=np.random.randint(60, 300))
        events.append(build_base_row(
            timestamp, 2, session_id, user_id,
            np.random.choice(COMPUTERS[20:]),   # servers
            "EventID_7.0", "denied_access",
            "power_user", source_ip, random_ip(),
            "SMB", np.random.randint(1000, 9000)
        ))

    # Stage 3: Credential exploitation (successful auth)
    timestamp += timedelta(seconds=np.random.randint(120, 600))
    events.append(build_base_row(
        timestamp, 2, session_id, user_id,
        np.random.choice(COMPUTERS[20:]),
        "EventID_9.0", "none",
        "administrator", source_ip, random_ip(),
        "KERBEROS", np.random.randint(1000, 9000)
    ))

    # Stage 4: Lateral movement (3-5 additional systems)
    n_lateral = np.random.randint(3, 6)
    for _ in range(n_lateral):
        timestamp += timedelta(minutes=np.random.randint(1, 8))
        events.append(build_base_row(
            timestamp, 2, session_id, user_id,
            np.random.choice(COMPUTERS),
            "EventID_3.0", "unauthorized_lateral_access",
            "administrator", source_ip, random_ip(),
            np.random.choice(["SMB", "RDP", "WMI"]),
            np.random.randint(1000, 9000)
        ))

    # Stage 5: Target access
    timestamp += timedelta(minutes=np.random.randint(5, 15))
    events.append(build_base_row(
        timestamp, 2, session_id, user_id,
        COMPUTERS[-1],                          # highest value server
        "EventID_13.0", "policy_bypass_attempt",
        "domain_admin", source_ip, random_ip(),
        "LDAP", np.random.randint(1000, 9000)
    ))

    return events


# ════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ════════════════════════════════════════════════════════════

def generate_dataset(output_path="data/zt_policy_violations.csv"):
    """
    Generate the full 6,016-event dataset and save to CSV.
    Matches Table 3.1 (dataset summary) and Table 4.1 (class distribution).
    """
    print("=" * 60)
    print("Generating Synthetic Zero Trust Policy Violation Dataset")
    print("=" * 60)

    base_date = datetime(2024, 1, 1)
    all_rows  = []
    session_counter = 1

    # ── Generate NORMAL events (5,800) ──
    print(f"[1/3] Generating {N_NORMAL} normal events...")
    for i in range(N_NORMAL):
        day     = np.random.randint(0, 365)
        date    = base_date + timedelta(days=day)
        user_id = f"USR-{np.random.randint(1, N_USERS + 1):03d}"
        sid     = f"SES-{session_counter:05d}"
        if i % 15 == 0:
            session_counter += 1
        all_rows.append(generate_normal_event(date, sid, user_id))

    # ── Generate SUSPICIOUS events (150) ──
    print(f"[2/3] Generating {N_SUSPICIOUS} suspicious events...")
    for i in range(N_SUSPICIOUS):
        day     = np.random.randint(0, 365)
        date    = base_date + timedelta(days=day)
        user_id = f"USR-{np.random.randint(1, N_USERS + 1):03d}"
        sid     = f"SES-{session_counter:05d}"
        if i % 3 == 0:
            session_counter += 1
        all_rows.append(generate_suspicious_event(date, sid, user_id))

    # ── Generate MALICIOUS sequences (66 events total) ──
    print(f"[3/3] Generating {N_MALICIOUS} malicious events...")
    malicious_generated = 0
    while malicious_generated < N_MALICIOUS:
        day     = np.random.randint(0, 365)
        date    = base_date + timedelta(days=day)
        user_id = f"ATK-{np.random.randint(1, 6):03d}"
        sid     = f"SES-{session_counter:05d}"
        session_counter += 1

        seq = generate_malicious_sequence(date, sid, user_id)

        # Only take as many as we still need
        remaining = N_MALICIOUS - malicious_generated
        seq       = seq[:remaining]
        all_rows.extend(seq)
        malicious_generated += len(seq)

    # ── Shuffle to mix classes realistically ──
    np.random.shuffle(all_rows)

    # ── Build DataFrame ──
    df = pd.DataFrame(all_rows)

    # ── Verify counts ──
    print("\n── Dataset Summary ──")
    print(f"Total events : {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nClass distribution:")
    dist = df["Label"].value_counts().sort_index()
    labels = {0: "Normal", 1: "Suspicious", 2: "Malicious"}
    for lbl, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {labels[lbl]:12s} (Label {lbl}): {count:5d}  ({pct:.1f}%)")

    # ── Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")
    print("=" * 60)

    return df


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 3 rows preview:")
    print(df[["SystemTime", "SystemTime_hour", "Label",
              "violation_type", "privilege_level"]].head(3))