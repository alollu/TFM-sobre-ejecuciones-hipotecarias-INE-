# Data Dictionary

**Archivo original**: data/raw/ejecuciones_hipotecarias.csv

| Columna | Tipo | Unidad | Descripción | Notas de limpieza |
|---|---:|---|---|---|
| ccaa | string | - | Comunidad Autónoma | Normalizar nombres; eliminar tildes |
| year | int | año | Año de referencia | Convertir a int |
| total | float | fincas | Total ejecuciones | Revisar nulos |
| persona_fisica | float | fincas | Viviendas con titular persona física |  |
| persona_juridica | float | fincas | Viviendas con titular persona jurídica |  |
