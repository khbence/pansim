# HCW-intervenciós kísérlet - eredmények és kiértékelés (magyar összefoglaló)

Ez a dokumentum a `README.md` (angol, technikai leírás - futtatás, fájlstruktúra, pontos
parancssori kapcsolók) eredményeinek és azok kiértékelésének magyar nyelvű összefoglalója. A
futtatási/telepítési részletekért lásd az angol verziót; itt kizárólag a kutatási kérdésre, a
módszertanra és az eredményekre koncentrálunk.

## A kutatási kérdés

Az egészségügyi dolgozók (healthcare workers, HCW) a lakosság egy speciális, potenciálisan
veszélyeztetett (vulnerable) csoportja egy járvány alatt, mivel munkájuk során fokozottan
érintkeznek fertőzött betegekkel. A kísérlet célja két dolog:

1. **Igazolható-e statisztikailag, hogy a HCW csoport ténylegesen sérülékenyebb**, mint a
   lakosság többi része?
2. **Mely NPI-k (nem-gyógyszeres beavatkozások: karantén, zárlatok, tesztelés) és az oltási
   politika hogyan befolyásolják ezt a sérülékenységet** - melyik a legfontosabb, és hogyan hatnak
   egymásra?

## Módszertan röviden

### Alapfogalmak

- **HCI** (panSim kimeneti oszlop): a HCW-k hány százaléka fertőzött VAGY karanténban az adott
  napon (prevalencia, egész számra kerekítve).
- **HCE** (panSim kimeneti oszlop): hány HCW fertőződött meg újonnan az elmúlt 24 órában
  (incidencia, pontos darabszám).
- **HCW definíció**: olyan ügynök, akinek munkahelye kórház (típus 12) vagy orvosi rendelő
  (típus 14) - lásd `flagHealthcareWorkers()`, `src/runtime/simulationOrchestration.cpp:355-389`.

### A kritikus módszertani javítás: a populáció nem tartalmazhatja a HC-t

Az első próbálkozás során a HCI/HCE értékeket közvetlenül a **teljes populáció** (mindenki,
HCW-kkel együtt) megfelelő értékeihez hasonlítottuk - ez hibás, mert a HCW csoport a teljes
populáció egy **része**, nem egy attól független csoport. A helyes összehasonlításhoz ki kellett
vonni a HCW-k hozzájárulását a populációs számokból, hogy egy valódi, át nem fedő "a lakosság
többi része" csoportot kapjunk.

Ehhez pontosan ki kellett számolni:
- **HEALTHCARE_WORKER_COUNT = 10 136** - a valós input fájlokból (`agents1.json` +
  `locations0.json`) számolva, pontosan reprodukálva a C++ `flagHealthcareWorkers()` logikáját.
- **TOTAL_POPULATION = 178 746** (a 184 745 ügynökből kivonva az 5999 "turistát", mivel a
  `--disableTourists` alapértelmezetten be van kapcsolva, és ők sosem lépnek be a szimulációba).
- HCW-k aránya: **5,67%** a teljes populációból.

### A végső mérőszám: relatív kockázat (RR)

Minden futásra kiszámoltuk a HCW csoport és "a lakosság többi része" (HCW-k nélkül) napi
**fejenkénti** (per-capita) mutatóit, majd ezek arányát:

- **`incidence_RR`** = (HCW új fertőzési ráta) / (lakosság többi részének új fertőzési rátája).
  Mindkét oldal **pontos** szám (HCE, illetve NI-HCE), nincs benne becslés vagy torzítás - **ez a
  megbízható mérőszám**.
- **`prevalence_RR`** = (HCW fertőzött-vagy-karanténban arány) / (lakosság többi részének
  ugyanilyen aránya). Mivel a HCI definíciója kombinálja a fertőzést és a karantént, ez a mutató
  **torzulhat egy karanténpolitika-változástól** anélkül, hogy a tényleges fertőzési kockázat
  változna - lásd lentebb.

RR > 1 azt jelenti, hogy a HCW csoport arányosan jobban érintett, mint a lakosság többi része;
RR = 1 azt, hogy nincs különbség; RR < 1 azt, hogy a HCW csoport kevésbé érintett.

Minden forgatókönyvet **20 független szimulációs futással** teszteltünk, és a 20 vs. 20 futás
közötti különbséget kétmintás Mann-Whitney U-teszttel vizsgáltuk (Benjamini-Hochberg-korrekcióval
a többszörös összehasonlításra).

## Tesztelt forgatókönyvek

| forgatókönyv | mit változtat a `baseline`-hoz képest |
|---|---|
| `baseline` | validált alapbeállítás (karantén szint 3, teljes zárlat-ütemterv, alapértelmezett oltási prioritás és tesztelési ráta) |
| `quarantine0` | karantén teljesen kikapcsolva (`--quarantinePolicy 0`) |
| `closuresNone` | összes zárlat/korlátozás kikapcsolva (üres szabálylista) |
| `noVacc` | oltás teljesen kikapcsolva, mindenkinek |
| `noVaccHCW` | **csak a HCW-k** vannak kizárva az oltásból, a lakosság többi része változatlanul kap oltást |
| `hiTest` | kórházi tesztelési ráta megemelve 0,005-ről 0,5-re |
| `noVaccHCW_hiTest` | `noVaccHCW` + `hiTest` kombinációja |
| `closuresNone_hiTest` | `closuresNone` + `hiTest` kombinációja |
| `quarantine0_hiTest` | `quarantine0` + `hiTest` kombinációja |
| `quarantine0_noVaccHCW` | `quarantine0` + `noVaccHCW` kombinációja (a két "káros" tényező együtt) |

## Fő eredmény: a HCW csoport valóban sérülékeny - minden körülmény között

Mind a 10 forgatókönyv mind a 200 futásában (10 szcenárió × 20 futás), **kivétel nélkül**,
`incidence_RR > 1`. Minden egyes szcenárió statisztikailag szignifikánsan különbözik az RR=1
nullhipotézistől (Wilcoxon előjeles rangteszt, p ≈ 1,9×10⁻⁶ minden esetben - ez a lehető
legmagasabb szignifikancia 20 mintával). Tehát: **az egészségügyi dolgozók minden tesztelt
beavatkozási környezetben magasabb napi fertőzési rátával rendelkeznek, mint a lakosság többi
része, kivétel nélkül.**

## Teljes eredménytábla

| forgatókönyv | `incidence_RR` | különbség a baseline-hoz képest | szignifikáns? |
|---|---|---|---|
| baseline | 1,38 | - | - |
| closuresNone | 1,38 | +0,00 | nem |
| hiTest | 1,09 | -0,29 | igen |
| closuresNone_hiTest | 1,13 | -0,25 | igen |
| quarantine0 | 1,47 | +0,09 | igen |
| quarantine0_hiTest | 1,48 | +0,10 | igen |
| noVacc | 1,78 | +0,40 | igen |
| noVaccHCW | 1,82 | +0,44 | igen |
| noVaccHCW_hiTest | 1,52 | +0,14 | igen |
| quarantine0_noVaccHCW | 1,62 | +0,25 | igen |

(`outputs/relative_risk_table.csv` - a `prevalence_RR` oszlop is elérhető ugyanott, de lásd lent
az óvatosságra intő megjegyzést.)

## Részletes kiértékelés

### 1. Miért viselkedik eltérően a `prevalence_RR` és az `incidence_RR`? - a kórházak ki vannak véve a munkahelyi karanténból

A `quarantineAgent()` függvény (`include/policies/movementPolicies.h:179-248`) szintenként
súlyosbítja a karantént egy diagnosztizált ügynöknél:

```
szint 0        -> semmi
szint 1+       -> maga az ügynök karanténba kerül
szint 2+       -> + az OTTHONA is karanténba kerül
szint 3+       -> + a MUNKAHELYE/OSZTÁLYTERME is karanténba kerül - DE lásd lent
```

A munkahelyi karantén lépés pontosan ellenőrzi, hogy a hozzárendelt helyszín **valódi típusa**
(nem csak az, hogy ez a "munka" időrés) pontosan "munkahely" (4) vagy "osztályterem" (33)-e. Egy
HCW esetén a munkahelyéhez tartozó helyszín valódi típusa 12 (kórház) vagy 14 (orvosi rendelő) -
pontosan ugyanaz a feltétel, amivel a rendszer egyáltalán HCW-nek minősíti őt. Emiatt ez a feltétel
**sosem igaz** egy HCW munkahelyére, tehát **a kórházak/rendelők sosem kerülnek karanténba/bezárásra**
ezen a csatornán, semmilyen karantén-szinten. A kód saját kikommentezett sorai is megerősítik, hogy
ez szándékos: a kórházaknak működniük kell járvány alatt.

**Következmény**: a `baseline` (szint 3) alatt egy átlagos munkahely rendszeresen "bezár"
(karanténba kerül), amikor egy beteg munkatársat találnak ott - ez sok kollégát küld egyszerre
karanténba. A kórházak ezt sosem teszik a saját dolgozóikkal. Emiatt a `prevalence_RR` (ami a
"fertőzött VAGY karanténban" definíciót örökli mindkét oldalon) erősen torzul a karanténpolitikától
függően (0,51 `baseline`-nál, de 1,14 `quarantine0` alatt!), míg az `incidence_RR` (tiszta
fertőzés, karantén nélkül) stabil marad - **ezért az `incidence_RR` a megbízható mutató**.

### 2. Az oltási prioritás izolálása (`noVaccHCW`)

A `noVacc` (mindenkinél kikapcsolt oltás) és a `noVaccHCW` (csak a HCW-knél kikapcsolt oltás)
összehasonlítása megválaszolja: a lakosság általános beoltottsága (közösségi/nyájimmunitás) vagy a
HCW-k saját oltási hozzáférése számít-e jobban?

**Eredmény**: `noVaccHCW` (1,82) enyhén, de szignifikánsan **rosszabb**, mint `noVacc` (1,78)
(különbség +0,046, p=0,0008). Tehát **nem elsősorban a lakosság nyájimmunitása védi a HCW-ket** -
a saját oltási hozzáférésük számít, és ennek elvesztését nem kompenzálja érdemben az, hogy a
körülöttük lévők be vannak oltva. Ez alátámasztja, hogy a HCW-k oltási prioritása önálló,
nélkülözhetetlen védelmi intézkedés, nem helyettesíthető populációs szintű lefedettséggel.

### 3. Pótolja-e a fokozott tesztelés az elvesztett oltási prioritást? (`noVaccHCW_hiTest`)

A két eddig legfontosabbnak talált tényező kombinációja. `incidence_RR`: 1,38 (baseline) → 1,82
(`noVaccHCW` egyedül, a legrosszabb egyedi tényező) → 1,09 (`hiTest` egyedül, az egyetlen védő
hatású tényező) → **1,52 (a kettő együtt)**.

Egy tisztán additív modell (nincs kölcsönhatás) ezt jósolná: 1,38 + 0,445 (`noVaccHCW` hatása) -
0,285 (`hiTest` hatása) = 1,538 - ez alig 0,02-del tér el a megfigyelt 1,52-től. **A két tényező
gyakorlatilag additív, nincs érdemi kölcsönhatás köztük** - két független, egymásra rakódó
védelmi/kockázati réteg, nem egymást helyettesítő mechanizmusok.

Számszerűen: az oltási prioritás elvesztése önmagában +0,445-tel növeli az RR-t; fokozott
teszteléssel ez a többlet +0,142-re csökken - **a tesztelés az oltási prioritás elvesztéséből
eredő többletkockázat kb. 68%-át pótolja, de nem az egészet**. (Megjegyzés: a tesztelési kapcsoló
csak a 0-258. napokig tér el a baseline-tól, mert az alapértelmezett zárlat-fájl saját maga
0,5-re állítja a kórházi tesztelést a 259. naptól - ez a 68%-os szám tehát elsősorban az első
hullámra vonatkozik.)

### 4. Tartja-e magát a tesztelési hatás torzítatlanul mérve? (`closuresNone_hiTest`)

Megpróbáltuk kiküszöbölni a fenti torzítást azzal, hogy a `closuresNone` forgatókönyvvel
eltávolítjuk a zárlat-fájl saját tesztelés-felülíró szabályát is - **ez azonban nem ad tiszta
mérést**, mert a `closuresNone` ezen felül **minden variáns-bevezetést is eltávolít** (nincs
Alpha/Delta/Omikron a szimulációban, csak az eredeti variáns fut egész évben) és minden más
zárlati szabályt is (maszk, kijárási tilalom, iskolai korlátozás). Ez sokkal nagyobb változás, mint
amit ki akartunk küszöbölni.

**Eredmény**: `closuresNone_hiTest` RR-je 1,13 (-0,25), nagyon közel `hiTest` önmagában mért
-0,29-es hatásához - megnyugtató (nem egy sokkal nagyobb, elrejtett hatásról van szó), de nem egy
tisztán izolált mérés, mert a variáns-bevezetés hiánya egybemosódik a tesztelés hatásával.

### 5. A legfontosabb mechanisztikus felfedezés: a tesztelés hatástalan karantén nélkül (`quarantine0_hiTest`)

Ez a három új forgatókönyv közül a legfontosabb eredmény. A `quarantine0_hiTest` RR-je (1,477)
statisztikailag **megkülönböztethetetlen** a puszta `quarantine0`-tól (1,469) - közvetlen
Mann-Whitney teszt, **p=0,51, nem szignifikáns**. Ez éles ellentétben áll azzal a kb. 68%-os
kártérítéssel, amit a tesztelés az oltás elvesztésénél nyújtott.

Ennek oka pontosan visszavezethető a kódra: a tesztelés védő hatásának teljes mechanizmusa
"észlelés, majd elkülönítés" - egy pozitív teszteredmény csak akkor számít, ha az illetőt
ténylegesen karanténba is helyezik emiatt. A `quarantineAgent()` függvény
(`include/policies/movementPolicies.h:181`) **azonnal visszatér**, ha `a.quarantinePolicy == 0`,
mielőtt bármi más történne - tehát `quarantine0` alatt egy pozitív teszt semmit nem változtat az
adott ügynök viselkedésén. **A tesztelésnek ebben a modellben nincs önálló,
fertőzés-csökkentő mechanizmusa - kizárólag a karantén-csatornán keresztül fejti ki hatását.**

**Ez újrarendezi az egész rangsort**: a tesztelés nem egy önálló, negyedik, egyenrangú "kar" a
karantén, az oltás és a zárlatok mellett - hanem egy **szorzó** a már meglévő karanténpolitikán. Az
"emeljük a tesztelést" csak akkor jó tanács, ha "és van karantén/elkülönítési infrastruktúra, ami
reagál az eredményre". A karantén saját, közvetlen hatása (+0,09 kikapcsolva) ezért alulbecsüli a
valódi fontosságát - egyben **előfeltétele** is annak, hogy a tesztelésnek egyáltalán legyen hatása.

### 6. A két káros tényező együtt: szub-additivitás és egy meglepő fordulat (`quarantine0_noVaccHCW`)

Additív predikció: 1,38 + 0,09 (`quarantine0`) + 0,45 (`noVaccHCW`) = 1,91. Megfigyelt: **1,62** -
nagy (-0,29) eltérés az additivitástól, éles ellentétben a `noVaccHCW_hiTest` szinte tökéletes
additivitásával (-0,02 eltérés). A két káros tényező **nem egyszerűen összeadódik**.

Közvetlen páronkénti tesztek derítik ki, miért:
- `noVaccHCW` hozzáadása `quarantine0`-hoz: RR +0,155-tel nő, szignifikáns - kisebb, mint a
  baseline-hoz hozzáadva mért +0,445, de még mindig egyértelműen káros irányban.
- `quarantine0` hozzáadása `noVaccHCW`-hez: RR **-0,199-del csökken**, szignifikáns - ez az
  **ellentétes előjel**, mint a `quarantine0` önmagában mért hatása (+0,09)!

Ez a fordulat ugyanarra a mechanizmusra vezethető vissza, mint a prevalencia/incidencia
ellentmondása: mivel a kórházak strukturálisan ki vannak véve a munkahelyi karanténból, a HCW-k
eleve nem sokat profitáltak ebből a védelmi csatornából - **a lakosság többi része viszont igen**.
A karantén eltávolítása (`quarantine0`) ezért arányosan **jobban** megnöveli a lakosság többi
részének fertőzési rátáját, mint a HCW-két: az alapul szolgáló abszolút rátákat vizsgálva,
`noVaccHCW`-ről `quarantine0_noVaccHCW`-re váltva a HCW-incidencia **24%-kal** nő (16,7 → 20,8),
míg a lakosság többi részének incidenciája **39%-kal** (152,3 → 212,4) - a nevező gyorsabban "utoléri"
a számlálót, ami szűkíti az arányt annak ellenére, hogy mindkét csoportban abszolút értékben nő a
fertőzöttek száma.

## Összegzés: rangsor a HCW-sérülékenység szempontjából

1. **Oltás (közvetlen HCW-hozzáférés)** - a legnagyobb és legrobusztusabb hatás minden tesztelt
   környezetben (+0,40-0,45, ha megszűnik), és az egyetlen tényező, amelyről bizonyítottan a HCW-k
   *saját* hozzáférése számít, nem csak a populációs szintű nyájimmunitás.
2. **Tesztelés, de csak akkor, ha van aktív karantén** - nagy és megbízható hatás, ha a
   karanténpolitika érvényben van (-0,29, az oltási prioritás elvesztésének kb. 68%-át pótolja),
   de **önmagában hatástalan** (p=0,51), ha a karantén ki van kapcsolva. Nem önálló beavatkozás -
   a karanténtól függ.
3. **Karanténpolitika** - szerény *közvetlen* hatás (+0,09 kikapcsolva), ami alulbecsüli a valódi
   fontosságát, mivel egyben előfeltétele is a tesztelés sokkal nagyobb hatásának. Emellett az a
   tényező, amelynek eltávolítása paradox módon *szűkíti* a HCW-populáció rést, ha az oltási
   prioritás már amúgy is elveszett - emlékeztető arra, hogy ezek a tényezők egymással is
   kölcsönhatásban állnak, nem csak egy rögzített alaphelyzethez képest.
4. **Zárlatok/korlátozások (maszk, kijárási tilalom, iskolai korlátozás, helyszín-bezárások)** -
   egyik tesztben sem volt mérhető hatása a HCW-vs-populáció *arányra*. Valószínűleg azért, mert a
   zárlatok általánosan csökkentik a terjedést, foglalkozástól függetlenül - mindkét csoportot
   együtt mozgatják, nem differenciáltan.

## Végkövetkeztetés

Az egészségügyi dolgozók **statisztikailag igazolhatóan, minden tesztelt körülmény között**
sérülékenyebb csoportnak bizonyultak, mint a lakosság többi része. A legfontosabb, egyértelműen
alátámasztott szakpolitikai üzenet: **az egészségügyi dolgozók oltási prioritása az egyetlen
önmagában is robusztus védelmi intézkedés** - elvesztését sem a lakosság nyájimmunitása, sem
(teljes mértékben) a fokozott tesztelés nem pótolja. A tesztelés ugyanakkor fontos *kiegészítő*
intézkedés, feltéve, hogy létezik karanténpolitika, amely érvényt szerez a pozitív
teszteredményeknek - önmagában, karantén nélkül, a modellben nincs hatása.
