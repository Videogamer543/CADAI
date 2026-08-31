import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { useAuth, useIsAdmin } from "@/lib/auth";

const ENGINE_URL = "https://cadai-282761536722.us-central1.run.app/";

type Run = {
  id: string;
  kind: string;
  label: string | null;
  status: string;
  duration_ms: number | null;
  created_at: string;
};

type Quota = { monthly_limit: number; used_this_period: number; period_start: string };

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Workspace — CADAI" },
      { name: "description", content: "Your CADAI workspace: launch analyses, review run history and monthly usage." },
      { property: "og:title", content: "Workspace — CADAI" },
      { property: "og:description", content: "Launch analyses and review your CADAI run history." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Dashboard,
});

const KINDS = [
  { id: "fem_2d", label: "2D stress analysis" },
  { id: "fem_3d", label: "3D stress analysis" },
  { id: "chat", label: "Grounded Q&A" },
  { id: "manufacturing", label: "Machinability check" },
];

function Dashboard() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const isAdmin = useIsAdmin(user?.id);
  const [runs, setRuns] = useState<Run[]>([]);
  const [quota, setQuota] = useState<Quota | null>(null);
  const [kind, setKind] = useState(KINDS[0]!.id);
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState(false);
  const [panel, setPanel] = useState(false);
  const [frameKey, setFrameKey] = useState(0);

  useEffect(() => {
    if (!loading && !user) navigate({ to: "/auth" });
  }, [loading, user, navigate]);

  const refresh = useCallback(async () => {
    if (!user) return;
    const [r, q] = await Promise.all([
      supabase
        .from("analysis_runs")
        .select("id,kind,label,status,duration_ms,created_at")
        .order("created_at", { ascending: false })
        .limit(25),
      supabase.from("usage_quotas").select("monthly_limit,used_this_period,period_start").maybeSingle(),
    ]);
    if (r.data) setRuns(r.data as Run[]);
    if (q.data) setQuota(q.data as Quota);
  }, [user]);

  useEffect(() => {
    if (!user) return;
    void refresh();
    void supabase.from("profiles").update({ last_seen_at: new Date().toISOString() }).eq("id", user.id);
  }, [user, refresh]);

  async function launch() {
    setBusy(true);
    try {
      const { error } = await supabase.rpc("record_analysis_run", {
        _kind: kind,
        _label: label || "",
        _status: "started",
        _duration_ms: 0,
        _error: "",
        _metadata: { source: "workspace" },
      } as never);
      if (error) throw error;
      setLabel("");
      await refresh();
      setFrameKey((k) => k + 1);
      setPanel(false);
      toast.success("Run logged — engine reloaded");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not start run");
    } finally {
      setBusy(false);
    }
  }

  if (loading || !user) {
    return <main className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">Loading…</main>;
  }

  const used = quota?.used_this_period ?? 0;
  const limit = quota?.monthly_limit ?? 0;
  const pct = limit ? Math.min(100, Math.round((used / limit) * 100)) : 0;

  return (
    <main className="flex h-screen flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border">
        <div className="flex items-center justify-between gap-4 px-5 py-2.5">
          <div className="flex items-center gap-4">
            <Link to="/" className="font-mono text-sm tracking-[0.3em]">CADAI</Link>
            <span className="hidden font-mono text-xs text-muted-foreground sm:inline">
              {used}/{limit} runs this month
            </span>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <button
              onClick={() => setPanel((p) => !p)}
              className="rounded-md border border-border px-3 py-1.5 hover:bg-accent"
            >
              {panel ? "Close panel" : "Runs & usage"}
            </button>
            <button
              onClick={() => setFrameKey((k) => k + 1)}
              className="rounded-md border border-border px-3 py-1.5 hover:bg-accent"
            >
              Reload engine
            </button>
            {isAdmin && (
              <Link to="/admin" className="text-muted-foreground hover:text-foreground">Admin</Link>
            )}
            <span className="hidden text-muted-foreground lg:inline">{user.email}</span>
            <button
              onClick={async () => {
                await supabase.auth.signOut();
                navigate({ to: "/" });
              }}
              className="rounded-md border border-border px-3 py-1.5 hover:bg-accent"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1">
        <iframe
          key={frameKey}
          src={ENGINE_URL}
          title="CADAI engine"
          className="h-full w-full border-0 bg-background"
          allow="clipboard-read; clipboard-write; fullscreen"
        />

        {panel && (
          <aside className="absolute right-0 top-0 z-10 flex h-full w-full max-w-sm flex-col gap-6 overflow-y-auto border-l border-border bg-card p-5 shadow-2xl">
            <div>
              <h2 className="label-caps">Monthly usage</h2>
              <p className="mt-3 font-mono text-3xl">
                {used}
                <span className="text-lg text-muted-foreground"> / {limit}</span>
              </p>
              <div className="mt-3 h-1.5 w-full rounded-full bg-muted">
                <div className="h-1.5 rounded-full bg-primary" style={{ width: `${pct}%` }} />
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                Period started {quota ? new Date(quota.period_start).toLocaleDateString() : "—"}
              </p>
            </div>

            <div className="space-y-3">
              <h2 className="label-caps">Log a run</h2>
              <div>
                <label className="label-caps" htmlFor="kind">Analysis type</label>
                <select
                  id="kind"
                  value={kind}
                  onChange={(e) => setKind(e.target.value)}
                  className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  {KINDS.map((k) => (
                    <option key={k.id} value={k.id}>{k.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="label-caps" htmlFor="label">Part / job label</label>
                <input
                  id="label"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="bracket-rev-c"
                  className="mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-ring"
                />
              </div>
              <button
                onClick={launch}
                disabled={busy}
                className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
              >
                {busy ? "Logging…" : "Log run & reload engine"}
              </button>
              <a
                href={ENGINE_URL}
                target="_blank"
                rel="noopener"
                className="block text-center text-xs text-muted-foreground underline hover:text-foreground"
              >
                Open engine in a new tab
              </a>
            </div>

            <div>
              <h2 className="label-caps">Recent activity</h2>
              <ul className="mt-3 space-y-2 text-sm">
                {runs.length === 0 && <li className="text-muted-foreground">No runs yet.</li>}
                {runs.map((r) => (
                  <li key={r.id} className="rounded-md border border-border px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span>{KINDS.find((k) => k.id === r.kind)?.label ?? r.kind}</span>
                      <span className="font-mono text-xs text-muted-foreground">{r.status}</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                      <span>{r.label ?? "—"}</span>
                      <span className="font-mono">{new Date(r.created_at).toLocaleString()}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        )}
      </div>
    </main>
  );
}

