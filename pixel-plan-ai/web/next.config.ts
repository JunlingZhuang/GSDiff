import type { NextConfig } from "next";

const BACKEND = process.env.PIXEL_PLAN_BACKEND ?? "http://127.0.0.1:8765";
// Trace mode reaches the hfagent tracer through this rewrite; when the service
// is offline the studio degrades to the paste/upload seed path (no coupling).
const TRACER = process.env.HFAGENT_TRACER ?? "http://127.0.0.1:8801";

const nextConfig: NextConfig = {
  experimental: {
    // 2026-07-09 incident: a 24-room generation exceeded Next's default proxy
    // timeout (30s) mid-POST, so the rewrite to the Python backend returned a
    // plain-text "Internal Server Error" while the backend worker kept running
    // fine. Raise the rewrite/proxy timeout to 600s (value is in ms) so long
    // generations stream through. Read at config.experimental.proxyTimeout in
    // next/dist/server/lib/router-server.js.
    proxyTimeout: 600_000,
  },
  async rewrites() {
    // All plan computation lives in the Python backend; Next.js is UI only.
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/trace-api/:path*", destination: `${TRACER}/:path*` },
    ];
  },
};

export default nextConfig;
