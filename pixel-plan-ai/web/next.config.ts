import type { NextConfig } from "next";

const BACKEND = process.env.PIXEL_PLAN_BACKEND ?? "http://127.0.0.1:8765";

const nextConfig: NextConfig = {
  async rewrites() {
    // All plan computation lives in the Python backend; Next.js is UI only.
    return [{ source: "/api/:path*", destination: `${BACKEND}/api/:path*` }];
  },
};

export default nextConfig;
