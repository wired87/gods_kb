"""
Workflow step extracted from ``uniprot_kb.UniprotKB`` for ``main``.

CHAR: runs in-process on the same ``UniprotKB`` instance (``self``); keep signatures aligned
with the class delegator in ``uniprot_kb.py``.
"""
from __future__ import annotations

import asyncio


async def enrich_bioelectric_properties(self):
    """
    Integriert biophysikalische Parameter (Ionenselektivität, Leitfähigkeit,
    Spannungsabhängigkeit) für Proteine via GtoPdb.
    Pfad: PROTEIN -> ELECTRICAL_COMPONENT
    """
    _GTOP_BASE = "https://www.guidetopharmacology.org/services"
    protein_nodes = [(k, v) for k, v in self.g.G.nodes(data=True)
                     if v.get("type") == "PROTEIN" and self._is_active(k)]

    for node_id, protein in protein_nodes:
        uniprot_acc = protein.get("id")
        if not uniprot_acc:
            continue

        try:
            # A: GtoPdb Target-ID via UniProt Accession
            lookup_url = f"{_GTOP_BASE}/targets?accession={uniprot_acc}&database=UniProt"
            lookup_res = await self.client.get(lookup_url)
            if lookup_res.status_code != 200:
                continue
            targets = lookup_res.json()
            if not targets:
                continue

            target = targets[0]
            target_id = target.get("targetId")
            target_class = target.get("type", "UNKNOWN")
            if not target_id:
                continue

            # B: Drei biophysikalische Endpunkte parallel
            sel_url = f"{_GTOP_BASE}/targets/{target_id}/ionSelectivity"
            cond_url = f"{_GTOP_BASE}/targets/{target_id}/ionConductance"
            volt_url = f"{_GTOP_BASE}/targets/{target_id}/voltageDependence"

            sel_task = self.client.get(sel_url)
            cond_task = self.client.get(cond_url)
            volt_task = self.client.get(volt_url)
            sel_res, cond_res, volt_res = await asyncio.gather(
                sel_task, cond_task, volt_task, return_exceptions=True,
            )

            # C: Daten extrahieren (leere Responses = kein Ionenkanal)
            ion_selectivity = []
            if not isinstance(sel_res, Exception) and sel_res.status_code == 200:
                sel_data = sel_res.json()
                if sel_data:
                    ion_selectivity = [
                        entry.get("ion", entry.get("species", ""))
                        for entry in (sel_data if isinstance(sel_data, list) else [sel_data])
                        if entry
                    ]

            conductance_pS = None
            if not isinstance(cond_res, Exception) and cond_res.status_code == 200:
                cond_data = cond_res.json()
                if cond_data:
                    first_cond = cond_data[0] if isinstance(cond_data, list) else cond_data
                    conductance_pS = first_cond.get("conductance") or first_cond.get("value")

            v_half = None
            slope_factor = None
            if not isinstance(volt_res, Exception) and volt_res.status_code == 200:
                volt_data = volt_res.json()
                if volt_data:
                    first_volt = volt_data[0] if isinstance(volt_data, list) else volt_data
                    v_half = first_volt.get("vHalf") or first_volt.get("v_half")
                    slope_factor = first_volt.get("slopeFactor") or first_volt.get("slope")

            # D: ELECTRICAL_COMPONENT nur erstellen wenn Daten vorhanden
            if not (ion_selectivity or conductance_pS is not None or v_half is not None):
                continue

            biophys_id = f"BIOPHYS_{uniprot_acc}"
            self.g.add_node({
                "id": biophys_id,
                "type": "ELECTRICAL_COMPONENT",
                "label": f"Circuit_{target.get('name', uniprot_acc)}",
                "target_class": target_class,
                "ion_selectivity": ion_selectivity,
                "conductance_pS": conductance_pS,
                "v_half_activation": v_half,
                "slope_factor": slope_factor,
                "species": "Human",
            })

            self.g.add_edge(
                src=node_id,
                trgt=biophys_id,
                attrs={
                    "rel": "DESCRIBED_AS_COMPONENT",
                    "src_layer": "PROTEIN",
                    "trgt_layer": "BIOELECTRIC",
                },
            )

            print(f"Bioelectric Enriched: {protein.get('label')} -> {target_class} "
                  f"(ions={ion_selectivity}, g={conductance_pS}pS, V½={v_half})")

        except Exception as e:
            print(f"Bioelectric Error for {protein.get('label')}: {e}")

# --- SÄULE 3: MIKROBIOM-METABOLISMUS-ACHSE (VMH) ───────────────
