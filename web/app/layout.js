import "./globals.css";

export const metadata = {
  title: "Celtics Live Win Probability",
  description:
    "Real-time win probability for 800 Boston Celtics games, 2016-17 to " +
    "2025-26. MSBA Directed Research, Brandeis University.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
