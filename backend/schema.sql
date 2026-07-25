create extension if not exists pgcrypto;

create table if not exists reports (
  id uuid primary key default gen_random_uuid(),
  market text not null,
  latitude double precision not null,
  longitude double precision not null,
  note text,
  image_url text,
  status text not null default 'pending'
    check (status in ('pending', 'analyzed', 'needs_review', 'reviewed', 'failed')),
  created_at timestamptz not null default now(),
  reviewed_at timestamptz
);

create table if not exists predictions (
  id uuid primary key default gen_random_uuid(),
  report_id uuid not null references reports(id) on delete cascade,
  model_version text not null,
  classes_json jsonb not null default '[]'::jsonb,
  max_confidence double precision not null check (max_confidence >= 0 and max_confidence <= 1),
  severity_score integer not null check (severity_score >= 0 and severity_score <= 100),
  risk_proxy integer not null,
  priority text not null check (priority in ('low', 'medium', 'high', 'urgent')),
  explanation text not null,
  inference_ms integer not null default 0,
  created_at timestamptz not null default now()
);

create table if not exists review_events (
  id uuid primary key default gen_random_uuid(),
  report_id uuid not null references reports(id) on delete cascade,
  old_status text,
  new_status text not null,
  reviewer_alias text,
  created_at timestamptz not null default now()
);

create index if not exists reports_created_at_idx on reports(created_at desc);
create index if not exists reports_status_idx on reports(status);
create index if not exists reports_market_idx on reports(market);
create index if not exists predictions_report_id_idx on predictions(report_id);
create index if not exists predictions_priority_idx on predictions(priority);

insert into storage.buckets (id, name, public)
values ('reports', 'reports', true)
on conflict (id) do update set public = excluded.public;
