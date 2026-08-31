import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAuth, useIsAdmin } from "@/lib/auth";

type Profile = {
  id: string;
  email: string | null;
  display_name: string | null;
  organization: string | null;
  last_seen_at: string;
  created_at: string;
};

type Run = {
  id: string;
  user_id: string;
  kind: string;
  label: string | null;
  status: string;
  created_at: string;
};

export const Route = createFileRoute("/admin")({
  head: () => ({
    meta: [
      { title: "Admin — CADAI" },
      { name: "description", content: "Admin overview of CADAI users, analysis runs and usage across the platform." },
      { property: "og:title", content: "Admin — CADAI" },
      { property: "og:description", content: "Users, runs and usage across CADAI." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Admin,
});

function Admin() {
  const { user, loading } = useAuth();
  const isAdmin = useIsAdmin(user?.id);
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!loading && !user) navigate({ to: "/auth" });
  }, [loading, user, navigate]);

  useEffect(() => {
    if (!user || !isAdmin) return;
    void (async () => {
      const [p, r] = await Promise.all([
        supabase.from("profiles").select("*").order("created_at", { ascending: false }).limit(200),
        supabase
          .from("analysis_runs")
          .select("id,user_id,kind,label,status,created_at")
          .order("created_at", { ascending: false })
          .limit(100),
      ]);
      if (p.data) setProfiles(p.data as Profile[]);
      if (r.data) setRuns(r.data as Run[]);
      setChecked(true);
    })();
  }, [user, isAdmin]);

  if (loading || !user) {
    return <main className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">Loading…</main>;
  }

  if (!isAdmin) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-3 px-6 text-center">
        <h1 className="text-lg font-semibold">Admin access required</h1>
        <p className="text-sm text-muted-foreground">This account doesn't have the admin role.</p>
        <Link to="/dashboard" className="rounded-md border border-border px-4 py-2 text-sm hover:bg-accent">
          Back to workspace
        </Link>
      </main>
    );
  }

  const nameFor = (id: string) =>
    profiles.find((p) => p.id === id)?.email ?? id.slice(0, 8);

  return (
    <main className="min-h-screen">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="font-mono text-sm tracking-[0.3em]">CADAI</Link>
          <Link to="/dashboard" className="text-sm text-muted-foreground hover:text-foreground">
            Workspace
          </Link>
        </div>
      </header>

      <div className="mx-auto max-w-6xl space-y-8 px-6 py-10">
        <div className="grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-3">
          <Stat label="Users" value={profiles.length} />
          <Stat label="Runs (last 100)" value={runs.length} />
          <Stat
            label="Active last 7 days"
            value={
              profiles.filter(
                (p) => Date.now() - new Date(p.last_seen_at).getTime() < 7 * 864e5,
              ).length
            }
          />
        </div>

        <section>
          <h2 className="label-caps">Users</h2>
          <Table
            head={["Email", "Name", "Joined", "Last seen"]}
            rows={profiles.map((p) => [
              p.email ?? "—",
              p.display_name ?? "—",
              new Date(p.created_at).toLocaleDateString(),
              new Date(p.last_seen_at).toLocaleString(),
            ])}
            empty={checked ? "No users yet." : "Loading…"}
          />
        </section>

        <section>
          <h2 className="label-caps">All analysis runs</h2>
          <Table
            head={["When", "User", "Type", "Label", "Status"]}
            rows={runs.map((r) => [
              new Date(r.created_at).toLocaleString(),
              nameFor(r.user_id),
              r.kind,
              r.label ?? "—",
              r.status,
            ])}
            empty={checked ? "No runs yet." : "Loading…"}
          />
        </section>
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-card p-6">
      <p className="label-caps">{label}</p>
      <p className="mt-2 font-mono text-3xl">{value}</p>
    </div>
  );
}

function Table({ head, rows, empty }: { head: string[]; rows: string[][]; empty: string }) {
  return (
    <div className="mt-3 overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted text-left text-xs text-muted-foreground">
          <tr>
            {head.map((h) => (
              <th key={h} className="px-4 py-2 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr><td colSpan={head.length} className="px-4 py-8 text-center text-muted-foreground">{empty}</td></tr>
          )}
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-border">
              {row.map((cell, j) => (
                <td key={j} className={j === 0 ? "px-4 py-2 font-mono text-xs text-muted-foreground" : "px-4 py-2"}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
