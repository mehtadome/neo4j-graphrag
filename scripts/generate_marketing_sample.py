#!/usr/bin/env python3
"""Generate synthetic relational marketing rows for graph / GraphRAG demos."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROWS = 500
SEED = 42
OUT = Path(__file__).resolve().parent.parent / "data" / "marketing_interactions.csv"


def main() -> None:
    random.seed(SEED)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    n_customers = 95
    n_campaigns = 32
    n_products = 28
    n_creatives = 45
    n_agencies = 6

    customers = [f"CUST-{i:04d}" for i in range(1, n_customers + 1)]
    campaigns = [f"CAMP-{i:03d}" for i in range(1, n_campaigns + 1)]
    products = [f"SKU-{i:04d}" for i in range(1, n_products + 1)]
    creatives = [f"CR-{i:04d}" for i in range(1, n_creatives + 1)]
    agencies = [f"AGY-{i:02d}" for i in range(1, n_agencies + 1)]

    channels = [
        "paid_search",
        "social_paid",
        "email",
        "webinar",
        "partner",
        "display",
        "organic_social",
        "content_syndication",
    ]
    event_types = [
        ("impression", 0.38),
        ("click", 0.22),
        ("landing_view", 0.12),
        ("form_submit", 0.08),
        ("mql", 0.06),
        ("demo_booked", 0.05),
        ("trial_start", 0.05),
        ("purchase", 0.04),
    ]
    segments = ["Enterprise", "Mid-Market", "SMB", "Startup"]
    regions = ["NA", "EMEA", "APAC", "LATAM"]

    # Stable preferences per customer for relational clustering
    deal_pool = [f"DEAL-{i:04d}" for i in range(1000, 1120)]
    cust_segment = {c: random.choice(segments) for c in customers}
    cust_region = {c: random.choice(regions) for c in customers}
    cust_deals = {
        c: random.sample(deal_pool, k=random.randint(1, 2)) for c in customers
    }
    campaign_agency = {camp: random.choice(agencies) for camp in campaigns}
    campaign_product_bias = {camp: random.sample(products, k=random.randint(2, 6)) for camp in campaigns}

    def pick_event_type() -> str:
        r = random.random()
        acc = 0.0
        for name, w in event_types:
            acc += w
            if r <= acc:
                return name
        return event_types[-1][0]

    def spend_for(channel: str, event: str) -> float:
        base = {
            "paid_search": 2.5,
            "social_paid": 1.2,
            "email": 0.15,
            "webinar": 4.0,
            "partner": 3.0,
            "display": 0.8,
            "organic_social": 0.05,
            "content_syndication": 1.0,
        }[channel]
        mult = {"impression": 0.3, "click": 1.0, "landing_view": 0.4}.get(event, 0.0)
        if mult == 0.0 and event in ("form_submit", "mql", "demo_booked"):
            mult = 2.5 + random.random() * 4
        if event in ("trial_start", "purchase"):
            mult = 6.0 + random.random() * 10
        noise = random.uniform(0.85, 1.15)
        return round(base * mult * noise, 2)

    def revenue_for(event: str) -> float:
        if event == "purchase":
            return round(random.uniform(1200, 28000), 2)
        if event == "trial_start":
            return round(random.uniform(0, 500), 2)
        if event == "demo_booked":
            return round(random.uniform(0, 200), 2)
        return 0.0

    rows: list[dict[str, str | float]] = []
    for i in range(1, ROWS + 1):
        customer_id = random.choice(customers)
        campaign_id = random.choice(campaigns)
        channel = random.choice(channels)
        event = pick_event_type()
        session_id = f"SES-{random.randint(1, 220):05d}"
        product_pool = campaign_product_bias[campaign_id]
        product_id = random.choice(product_pool)
        creative_id = random.choice(creatives)
        agency_id = campaign_agency[campaign_id]
        segment = cust_segment[customer_id]
        region = cust_region[customer_id]

        offset_min = random.randint(0, 90 * 24 * 60)
        occurred_at = start + timedelta(minutes=offset_min)

        spend = spend_for(channel, event)
        revenue = revenue_for(event)
        deal_id = random.choice(cust_deals[customer_id])

        rows.append(
            {
                "interaction_id": f"INT-{i:05d}",
                "occurred_at": occurred_at.isoformat(),
                "customer_id": customer_id,
                "campaign_id": campaign_id,
                "product_id": product_id,
                "channel": channel,
                "event_type": event,
                "session_id": session_id,
                "region": region,
                "segment": segment,
                "spend_usd": spend,
                "revenue_usd": revenue,
                "creative_id": creative_id,
                "agency_id": agency_id,
                "deal_id": deal_id,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "interaction_id",
        "occurred_at",
        "customer_id",
        "campaign_id",
        "product_id",
        "channel",
        "event_type",
        "session_id",
        "region",
        "segment",
        "spend_usd",
        "revenue_usd",
        "creative_id",
        "agency_id",
        "deal_id",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
