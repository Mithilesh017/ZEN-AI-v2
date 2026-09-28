export type Me = {
  name: string;
  email: string;
  picture: string | null;
  display_name: string;
};

const redirectToLogin = () => {
  window.location.href = "/login";
};

export async function fetchMe(): Promise<Me | null> {
  const res = await fetch("/api/me");
  if (res.status === 401) {
    redirectToLogin();
    return null;
  }
  if (!res.ok) throw new Error(`Failed to load profile (${res.status})`);
  return res.json();
}

export async function saveDisplayName(displayName: string): Promise<string> {
  const res = await fetch("/api/display_name", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
  if (res.status === 401) redirectToLogin();
  if (!res.ok) throw new Error("Could not save your name");
  const data: { display_name: string } = await res.json();
  return data.display_name;
}
