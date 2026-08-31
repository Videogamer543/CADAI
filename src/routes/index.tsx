import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { getPublicStats } from "@/lib/stats.functions";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CADAI — AI-Assisted FEM & Manufacturing Analysis" },
      {
        name: "description",
        content:
          "CADAI runs cited finite-element stress analysis, meshing and machining checks on your CAD models. Sign in to run studies and track usage.",
      },
      { property: "og:title", content: "CADAI — AI-Assisted FEM & Manufacturing Analysis" },
      {
        property: "og:description",
        content:
          "Cited finite-element stress analysis, meshing and machining checks for your CAD models.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Landing,
});

const FEATURES = [
  {
    k: "01",
    t: "Finite element analysis",
    d: "2D and 3D stress fields, displacement and factor-of-safety on imported geometry.",
  },
  {
    k: "02",
    t: "Grounded engineering answers",
    d: "Every claim is traced back to a source in the knowledge base rather than guessed.",
  },
  {
    k: "03",
    t: "Materials & manufacturability",
    d: "Material library lookups, pocketing and machining feasibility checks.",
  },
  {
    k: "04",
    t: "Accounts & usage tracking",
    d: "Per-user history, monthly run quotas and an admin view of all activity.",
  },
];

function Landing() {
  const { user, loading } = useAuth();
  const [userCount, setUserCount] = useState<number | null>(null);

  useEffect(() => {
    getPublicStats().then((stats) => setUserCount(stats.userCount));
  }, []);

  return (
    <main className="min-h-screen">
      <header className="border-b border-border">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <span className="font-mono text-sm tracking-[0.3em] text-foreground">CADAI</span>
          <nav className="flex items-center gap-3 text-sm">
            {!loading && user ? (
              <Link
                to="/dashboard"
                className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90"
              >
                Dashboard
              </Link>
            ) : (
              <>
                <Link to="/auth" className="text-muted-foreground hover:text-foreground">
                  Sign in
                </Link>
                <Link
                  to="/auth"
                  search={{ mode: "signup" }}
                  className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90"
                >
                  Create account
                </Link>
              </>
            )}
          </nav>
        </div>
      </header>

      <section className="grid-backdrop border-b border-border">
        <div className="mx-auto max-w-5xl px-6 py-24">
          <p className="label-caps">Python FEM · Grounded AI · Manufacturing</p>
          <h1 className="mt-5 max-w-3xl text-4xl font-semibold leading-tight sm:text-6xl">
            Structural analysis that shows its work.
          </h1>
          <p className="mt-6 max-w-xl text-base text-muted-foreground">
            CADAI meshes your geometry, solves the stress field, and answers engineering questions
            with citations instead of vibes. Now with accounts, run history and usage limits.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <Link
              to="/auth"
              search={{ mode: "signup" }}
              className="rounded-md bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:opacity-90"
            >
              Get started
            </Link>
            <Link
              to="/dashboard"
              className="rounded-md border border-border px-5 py-2.5 text-sm font-medium hover:bg-accent"
            >
              Open workspace
            </Link>
          </div>
          {userCount !== null && (
            <p className="mt-8 font-mono text-xs tracking-[0.2em] text-muted-foreground">
              {userCount === 0
                ? "Be the first engineer to sign up"
                : `${userCount} engineer${userCount === 1 ? "" : "s"} signed up`}
            </p>
          )}
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-6 py-20">
        <h2 className="label-caps">Capabilities</h2>
        <div className="mt-8 grid gap-px overflow-hidden rounded-lg border border-border bg-border sm:grid-cols-2">
          {FEATURES.map((f) => (
            <article key={f.k} className="bg-card p-6">
              <span className="font-mono text-xs text-muted-foreground">{f.k}</span>
              <h3 className="mt-3 text-lg font-medium">{f.t}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{f.d}</p>
            </article>
          ))}
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-6 py-8 text-xs text-muted-foreground">
          <span className="font-mono tracking-[0.2em]">CADAI</span>
          <span>Engineering analysis, cited.</span>
        </div>
      </footer>
    </main>
  );
}
