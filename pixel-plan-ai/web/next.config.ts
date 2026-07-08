import type { NextConfig } from "next";

const BACKEND = process.env.PIXEL_PLAN_BACKEND ?? "http://127.0.0.1:8765";
// Trace mode reaches the hfagent tracer through this rewrite; when the service
// is offline the studio degrades to the paste/upload seed path (no coupling).
const TRACER = process.env.HFAGENT_TRACER ?? "http://127.0.0.1:8801";

const nextConfig: NextConfig = {
  async rewrites() {
    // All plan computation lives in the Python backend; Next.js is UI only.
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/trace-api/:path*", destination: `${TRACER}/:path*` },
    ];
  },
};

export default nextConfig;
