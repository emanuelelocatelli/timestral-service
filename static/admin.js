async function api(url, options) {
  const response = await fetch(url, options);
  if (response.status === 401) {
    window.location.href = "/admin/login";
    return null;
  }
  return response;
}

function row(cells) {
  const tr = document.createElement("tr");
  for (const cell of cells) tr.append(cell);
  return tr;
}

function textCell(value) {
  const td = document.createElement("td");
  td.textContent = value ?? "";
  return td;
}

async function loadStatus() {
  const response = await api("/admin/api/status");
  if (!response) return;
  const data = await response.json();
  const state = data.model_loaded ? "modello caricato" : "modello in caricamento";
  document.getElementById("status-line").textContent =
    `${data.model} · ${data.device} · ${data.torch_num_threads} thread · ${state}`;
}

async function loadIps() {
  const response = await api("/admin/api/ips");
  if (!response) return;
  const body = document.getElementById("ips");
  body.replaceChildren();
  for (const item of await response.json()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost";
    button.textContent = "Rimuovi";
    button.addEventListener("click", async () => {
      await api(`/admin/api/ips/${item.id}`, {method: "DELETE"});
      loadIps();
    });
    const action = document.createElement("td");
    action.append(button);
    body.append(row([textCell(item.ip_address), textCell(item.description), action]));
  }
}

async function loadTokens() {
  const response = await api("/admin/api/tokens");
  if (!response) return;
  const body = document.getElementById("tokens");
  body.replaceChildren();
  for (const item of await response.json()) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "ghost";
    toggle.textContent = item.is_active ? "Revoca" : "Riattiva";
    toggle.addEventListener("click", async () => {
      await api(`/admin/api/tokens/${item.id}`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({is_active: !item.is_active}),
      });
      loadTokens();
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ghost";
    remove.textContent = "Elimina";
    remove.addEventListener("click", async () => {
      await api(`/admin/api/tokens/${item.id}`, {method: "DELETE"});
      loadTokens();
    });
    const action = document.createElement("td");
    action.append(toggle, remove);
    body.append(row([
      textCell(item.name),
      textCell(item.token_prefix),
      textCell(item.is_active ? "attivo" : "revocato"),
      action,
    ]));
  }
}

document.getElementById("ip-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const response = await api("/admin/api/ips", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ip_address: form.ip_address.value, description: form.description.value}),
  });
  if (!response || !response.ok) return;
  form.reset();
  loadIps();
});

document.getElementById("token-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const response = await api("/admin/api/tokens", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name: form.name.value}),
  });
  if (!response || !response.ok) return;
  const created = await response.json();
  const fresh = document.getElementById("fresh-token");
  fresh.hidden = false;
  fresh.textContent = `${created.name}: ${created.token}`;
  form.reset();
  loadTokens();
});

document.getElementById("logout").addEventListener("click", async () => {
  await api("/admin/api/logout", {method: "POST"});
  window.location.href = "/admin/login";
});

loadStatus();
loadIps();
loadTokens();
