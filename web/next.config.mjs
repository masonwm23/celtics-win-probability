/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Headshots and team logos are served by the NBA CDN. They are loaded with a
  // plain <img> rather than next/image so that a missing headshot degrades to
  // an initials badge instead of throwing, which matters because roughly 1% of
  // playing time belongs to players whose photo may not exist.
};
export default nextConfig;
