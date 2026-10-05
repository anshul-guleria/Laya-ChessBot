import type { NextConfig } from "next";

/**
 * Rewrites `/api/*` to the FastAPI backend so the browser sees a single origin.
 *
 * Without this, every fetch from the page would need an absolute backend URL and the
 * backend would need CORS configured. One origin in dev and production is worth the
 * rewrite.
 */
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
