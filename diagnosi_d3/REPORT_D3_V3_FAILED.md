# Report D3 Fix V3 - FALLITO

## Diagnosi

Le metriche della V3 mostrano un degrado totale del grafo:
- **Nets**: 1 (prima 31)
- **Edges**: 5 (prima 32)
- **Components**: 49 (prima >150)
- **Wire clusters**: 0

### Cause Root
Il refactoring richiesto nella FASE 2 prevedeva di rendere `separate_wires` permissivo, passando la quasi totalità dei segmenti (tutti quelli non-curve) a `symbol_segs`, rimandando la separazione wire/symbol al post-clustering tramite la funzione `_classify_cluster_by_shape`.

Tuttavia, passare i fili assieme ai simboli a DBSCAN (single-linkage) ha provocato un fenomeno di **chaining globale**: i fili, toccando vari simboli sparsi per la pagina, hanno causato la fusione di decine di componenti distinti in un unico "blob" gigante. 
Infatti, il numero di componenti è sceso a soli 49 (da oltre 150), a indicare che molti simboli sono stati fusi nello stesso cluster. Poiché questi cluster giganti contengono forme (`shapes`), la classificazione shape-based li ha etichettati tutti come `symbol`, lasciando `wire_clusters` a 0.
Infine, la logica di `recover_absorbed_wires` non ha potuto estrarre nulla (ha recuperato 1 solo segmento) perché i segmenti fili, per quanto lunghi, non fuoriuscivano dalla bounding box immensa dell'intero cluster-blob.

### Conclusione
L'approccio "classificazione post-clustering basata su shape" ha il difetto strutturale di permettere a DBSCAN di fondere tutto. La separazione pre-clustering in `separate_wires` è *necessaria* per impedire il chaining tra i simboli. 
Il problema dei fili frammentati va risolto sistemando la logica del pre-clustering (ad esempio, abbassando la soglia length-based per i segmenti asse-allineati, o considerando wire anche i fili che toccano un altro filo già noto), senza però buttare tutto dentro DBSCAN.
