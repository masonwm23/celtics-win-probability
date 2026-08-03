import "./globals.css";

export const metadata = {
  title: "Celtics Live Win Probability",
  description:
    "Real-time win probability for 636 Boston Celtics games, 2016-17 to " +
    "2023-24. MSBA Directed Research, Brandeis University.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
