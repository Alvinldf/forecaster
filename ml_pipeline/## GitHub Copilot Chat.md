## GitHub Copilot Chat

- Extension Version: 0.22.4 (prod)
- VS Code: vscode/1.95.3
- OS: Mac

## Network

User Settings:
```json
  "github.copilot.advanced": {
    "debug.useElectronFetcher": true,
    "debug.useNodeFetcher": false
  }
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 140.82.113.6 (13 ms)
- DNS ipv6 Lookup: ::ffff:140.82.113.6 (10 ms)
- Electron Fetcher (configured): HTTP 200 (351 ms)
- Node Fetcher: HTTP 200 (331 ms)
- Helix Fetcher: HTTP 200 (419 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.112.22 (12 ms)
- DNS ipv6 Lookup: ::ffff:140.82.112.22 (17 ms)
- Electron Fetcher (configured): HTTP 200 (345 ms)
- Node Fetcher: HTTP 200 (351 ms)
- Helix Fetcher: HTTP 200 (353 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).