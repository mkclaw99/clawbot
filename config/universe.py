"""
German stock universe — DAX 40 + MDAX selection.

All tickers use XETRA notation (.DE suffix) as recognised by yfinance.
Company names are kept as comments for readability.

Sources:
  - DAX 40 composition: Deutsche Börse (SDAX/MDAX/DAX index group)
  - MDAX: 50 mid-cap German stocks; we include the 25 most liquid/relevant
  - ETFs: XETRA-listed index ETFs for sector/benchmark context
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# DAX 40 — Germany's blue-chip index
# ---------------------------------------------------------------------------

DAX40: list[str] = [
    "ADS.DE",   # Adidas
    "AIR.DE",   # Airbus
    "ALV.DE",   # Allianz
    "BAS.DE",   # BASF
    "BAYN.DE",  # Bayer
    "BEI.DE",   # Beiersdorf
    "BMW.DE",   # BMW
    "BNR.DE",   # Brenntag
    "CBK.DE",   # Commerzbank
    "CON.DE",   # Continental
    "1COV.DE",  # Covestro
    "DB1.DE",   # Deutsche Börse
    "DBK.DE",   # Deutsche Bank
    "DHL.DE",   # DHL Group (Deutsche Post)
    "DTE.DE",   # Deutsche Telekom
    "EOAN.DE",  # E.ON
    "ENR.DE",   # Siemens Energy
    "FME.DE",   # Fresenius Medical Care
    "FRE.DE",   # Fresenius SE
    "HEI.DE",   # Heidelberg Materials
    "HNR1.DE",  # Hannover Rück
    "IFX.DE",   # Infineon Technologies
    "MBG.DE",   # Mercedes-Benz
    "MRK.DE",   # Merck KGaA
    "MTX.DE",   # MTU Aero Engines
    "MUV2.DE",  # Munich Re (Münchener Rück)
    "P911.DE",  # Porsche AG
    "PAH3.DE",  # Porsche SE (Automobil Holding)
    "PUM.DE",   # Puma
    "QIA.DE",   # Qiagen
    "RHM.DE",   # Rheinmetall
    "RWE.DE",   # RWE
    "SAP.DE",   # SAP
    "SHL.DE",   # Siemens Healthineers
    "SIE.DE",   # Siemens
    "SRT3.DE",  # Sartorius (preferred shares)
    "SY1.DE",   # Symrise
    "VNA.DE",   # Vonovia
    "VOW3.DE",  # Volkswagen (preferred shares)
    "ZAL.DE",   # Zalando
]

# ---------------------------------------------------------------------------
# MDAX — 25 most liquid / strategically relevant mid-caps
# ---------------------------------------------------------------------------

MDAX_SELECTED: list[str] = [
    "AIXA.DE",  # AIXTRON
    "BC8.DE",   # Bechtle
    "BOSS.DE",  # Hugo Boss
    "EVD.DE",   # CTS Eventim
    "FNTN.DE",  # freenet AG
    "GXI.DE",   # Gerresheimer
    "HLAG.DE",  # Hapag-Lloyd
    "HOT.DE",   # Hochtief
    "LEG.DE",   # LEG Immobilien
    "NDX1.DE",  # Nordex
    "NEM.DE",   # Nemetschek
    "RAA.DE",   # RATIONAL AG
    "SDF.DE",   # K+S
    "TUI1.DE",  # TUI
    "UTDI.DE",  # United Internet
    "WAF.DE",   # Siltronic
    "WUW.DE",   # Wüstenrot & Württembergische
    "DWS.DE",   # DWS Group
    "DHER.DE",  # Delivery Hero
    "HFG.DE",   # HelloFresh
    "SGL.DE",   # SGL Carbon
    "TKA.DE",   # ThyssenKrupp
]

# ---------------------------------------------------------------------------
# High-volatility / speculative German stocks (replaces US "meme" universe)
# ---------------------------------------------------------------------------

HIGH_VOLATILITY: set[str] = {
    "TUI1.DE",   # TUI — travel & tourism, heavily cyclical
    "NDX1.DE",   # Nordex — wind energy, growth/volatile
    "DHER.DE",   # Delivery Hero — high-burn growth
    "ZAL.DE",    # Zalando — e-commerce, volatile
    "HFG.DE",    # HelloFresh — volatile meal-kit sector
    "TKA.DE",    # ThyssenKrupp — restructuring play
    "SGL.DE",    # SGL Carbon — small, volatile
    "VNA.DE",    # Vonovia — rate-sensitive RE
}

# ---------------------------------------------------------------------------
# German / European ETFs (XETRA-listed) — for sector + benchmark context
# ---------------------------------------------------------------------------

GERMAN_ETFS: list[str] = [
    "EXS1.DE",  # iShares Core DAX UCITS ETF  (tracks DAX 40)
    "EXSA.DE",  # iShares EURO STOXX 50 UCITS ETF
    "DBXD.DE",  # Xtrackers DAX UCITS ETF
    "EXSD.DE",  # iShares MDAX UCITS ETF
    "EXH1.DE",  # iShares STOXX Europe 600 UCITS ETF
    "EXV1.DE",  # iShares STOXX Europe 600 Automobiles & Parts
    "EXV4.DE",  # iShares STOXX Europe 600 Financial Services
    "EXV6.DE",  # iShares STOXX Europe 600 Health Care
    "EXV8.DE",  # iShares STOXX Europe 600 Technology
]

# ---------------------------------------------------------------------------
# Convenience aliases used by signal_aggregator and price_historian
# ---------------------------------------------------------------------------

# Default universe for signal generation (DAX + top MDAX)
DEFAULT_UNIVERSE: list[str] = DAX40 + MDAX_SELECTED

# Full universe for price history download
FULL_UNIVERSE: list[str] = sorted(set(DAX40 + MDAX_SELECTED + GERMAN_ETFS))

# Ticker → human-readable name (for display)
TICKER_NAMES: dict[str, str] = {
    "ADS.DE":   "Adidas",
    "AIR.DE":   "Airbus",
    "ALV.DE":   "Allianz",
    "BAS.DE":   "BASF",
    "BAYN.DE":  "Bayer",
    "BEI.DE":   "Beiersdorf",
    "BMW.DE":   "BMW",
    "BNR.DE":   "Brenntag",
    "CBK.DE":   "Commerzbank",
    "CON.DE":   "Continental",
    "1COV.DE":  "Covestro",
    "DB1.DE":   "Deutsche Börse",
    "DBK.DE":   "Deutsche Bank",
    "DHL.DE":   "DHL Group",
    "DTE.DE":   "Deutsche Telekom",
    "EOAN.DE":  "E.ON",
    "ENR.DE":   "Siemens Energy",
    "FME.DE":   "Fresenius Medical Care",
    "FRE.DE":   "Fresenius",
    "HEI.DE":   "Heidelberg Materials",
    "HNR1.DE":  "Hannover Rück",
    "IFX.DE":   "Infineon",
    "MBG.DE":   "Mercedes-Benz",
    "MRK.DE":   "Merck KGaA",
    "MTX.DE":   "MTU Aero Engines",
    "MUV2.DE":  "Munich Re",
    "P911.DE":  "Porsche AG",
    "PAH3.DE":  "Porsche SE",
    "PUM.DE":   "Puma",
    "QIA.DE":   "Qiagen",
    "RHM.DE":   "Rheinmetall",
    "RWE.DE":   "RWE",
    "SAP.DE":   "SAP",
    "SHL.DE":   "Siemens Healthineers",
    "SIE.DE":   "Siemens",
    "SRT3.DE":  "Sartorius",
    "SY1.DE":   "Symrise",
    "VNA.DE":   "Vonovia",
    "VOW3.DE":  "Volkswagen",
    "ZAL.DE":   "Zalando",
    "AIXA.DE":  "AIXTRON",
    "BC8.DE":   "Bechtle",
    "BOSS.DE":  "Hugo Boss",
    "EVD.DE":   "CTS Eventim",
    "FNTN.DE":  "freenet",
    "GXI.DE":   "Gerresheimer",
    "HLAG.DE":  "Hapag-Lloyd",
    "HOT.DE":   "Hochtief",
    "LEG.DE":   "LEG Immobilien",
    "NDX1.DE":  "Nordex",
    "NEM.DE":   "Nemetschek",
    "RAA.DE":   "RATIONAL",
    "SDF.DE":   "K+S",
    "TUI1.DE":  "TUI",
    "UTDI.DE":  "United Internet",
    "WAF.DE":   "Siltronic",
    "TKA.DE":   "ThyssenKrupp",
    "DHER.DE":  "Delivery Hero",
    "HFG.DE":   "HelloFresh",
    "SGL.DE":   "SGL Carbon",
    "DWS.DE":   "DWS Group",
}
