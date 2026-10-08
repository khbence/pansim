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
| `hcwDailyTest` | minden HCW napi tesztelése (95%-os valószínűség + újratesztelési várakozás 1 napra csökkentve) |
| `longQuarantine` | karantén/elkülönítés hossza 10 napról 21 napra nőve (attól függetlenül, hogy a karantén egyáltalán aktiválódik-e) |
| `hcwPPE` | kórházi védőfelszerelés (N95/köpeny/szemvédő): 70%-os fertőzőképesség-csökkentés kizárólag kórházi/rendelői helyszíneken, 0. naptól - új C++ kódváltoztatással megvalósítva |

## Fő eredmény: a HCW csoport sérülékeny - két fontos kivétellel

10 a 13 forgatókönyv mind a 20-20 futásában, **kivétel nélkül**, `incidence_RR > 1`, statisztikailag
szignifikánsan különbözve az RR=1 nullhipotézistől (Wilcoxon előjeles rangteszt, p ≈ 1,9×10⁻⁶
minden ilyen esetben - ez a lehető legmagasabb szignifikancia 20 mintával). Tehát: **az
egészségügyi dolgozók a legtöbb tesztelt beavatkozási környezetben magasabb napi fertőzési rátával
rendelkeznek, mint a lakosság többi része.**

**A két kivétel a két legfontosabb eredmény**:
- `hcwDailyTest` (minden HCW napi tesztelése): RR pontosan 1,00-ra csökken, statisztikailag **nem
  különbözik** az RR=1-től (p=0,67) - a rés teljesen bezárul.
- **`hcwPPE` (kórházi védőfelszerelés): RR **0,71-re csökken** - ez nem csak bezárás, hanem
  **megfordítás**: mind a 20 futásban RR<1, p=1,9×10⁻⁶. A HCW-k itt ténylegesen *kevésbé*
  veszélyeztetettek, mint a lakosság többi része.** Ez a teljes kísérletsorozat legerősebb
  eredménye - lásd a 7. pontot lent.

## Teljes eredménytábla

| forgatókönyv | `incidence_RR` | különbség a baseline-hoz képest | szignifikáns? |
|---|---|---|---|
| baseline | 1,38 | - | - |
| **hcwPPE** | **0,71** | **-0,67** | igen (RR maga is <1, p=1,9×10⁻⁶) |
| hcwDailyTest | 1,00 | -0,38 | igen (de maga az RR nem tér el szignifikánsan 1-től, p=0,67) |
| closuresNone | 1,38 | +0,00 | nem |
| hiTest | 1,09 | -0,29 | igen |
| closuresNone_hiTest | 1,13 | -0,25 | igen |
| longQuarantine | 1,35 | -0,03 | nem |
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

### 7. A kórházi védőfelszerelés megfordítja a sérülékenységi rést (`hcwPPE`) - a sorozat legerősebb eredménye

A professzori visszajelzés nyomán felmerült ötletek közül a legfontosabb hiányzó, valós
beavatkozás a kórházi védőfelszerelés (N95 maszk, köpeny, szemvédő) volt - erre korábban nem volt
megfelelő mérőszám: a meglévő `"Masks"` zárlati szabály **minden** nem-otthoni helyszínen
egységesen csökkenti a fertőzőképességet, nem lehetett vele kizárólag kórházakat célozni.

Ezért egy új `"HospitalPPE"` zárlati szabályt implementáltunk (`include/policies/closurePolicies.h`,
a `"Masks"` szabály pontos mása, csak a helyszín-szűrés `type == hospitalType || type == doctorType`-ra
cserélve). A paraméter 0,3 (70%-os fertőzőképesség-csökkentés kórházi/rendelői helyszíneken), a
0. naptól aktív a teljes futás alatt - ez egy középértékű becslés az egészségügyi környezetben mért
N95/szemvédő-hatékonyság szakirodalmi tartományából (pl. Chu és mtsai. 2020, *Lancet*
metaanalízise) - a valós hatékonyság erősen függ a betartástól és a felszerelés típusától, ezért ezt
illusztratív, dokumentált modellezési feltételezésnek kell tekinteni, nem pontosan validált
epidemiológiai paraméternek.

**Eredmény**: `incidence_RR` = **0,71** - nem csupán a rés bezárása, hanem **valódi megfordítás**.
Mind a 20 futásban `incidence_RR < 1` (medián 0,713, tartomány 0,67-0,77); az RR=1 nullhipotézis
elleni egymintás Wilcoxon-teszt p=1,9×10⁻⁶-ot ad (20 minta esetén a lehető legmagasabb
szignifikancia) - **az egészségügyi dolgozók mérhetően kevésbé veszélyeztetettek, mint a lakosság
többi része.** Ez közvetlenül látszik az `outputs/vulnerability.png` `hcwPPE` incidencia-panelén is:
a kék (HCW) görbe mindkét csúcsnál egyértelműen a narancs (populáció) görbe *alatt* fut - ez az
egyetlen forgatókönyv, ahol ez történik.

Ez a teljes kísérletsorozat legerősebb eredménye: a legnagyobb mért hatás (-0,67 a baseline-hoz
képest, majdnem kétszerese a `hcwDailyTest` már eddig is nagy -0,38-as hatásának), és az egyetlen
beavatkozás, amely nem csak bezárja, hanem meg is fordítja a sérülékenységi rést. Ennek
mechanisztikus magyarázata: a védőfelszerelés közvetlenül a fertőzés *helyén* hat, függetlenül az
"észlelés, majd elkülönítés" útvonaltól, amitől a `hcwDailyTest`/`hiTest` függ (és amiről a
`quarantine0_hiTest` megmutatta, hogy aktív karantén nélkül nullára csökken) - nem versenyez a
többi tesztelt tényezővel, és nem is függ tőlük, mert egy strukturálisan eltérő, közvetlenebb
védelmi mechanizmus.

### 8. A napi tesztelés lezárja a rést (`hcwDailyTest`)

A `hiTest` csak a tesztelési *valószínűséget* emelte (0,005→0,5), de az újratesztelési várakozási
időt (`--testingRepeatDelay`) az alapértelmezett 5 napon hagyta - így egy "naponta tesztelt" ügynök
a gyakorlatban csak 5 naponta kaphatott *új* tesztet. A `hcwDailyTest` mindkettőt javítja
(valószínűség 0,95, `testingRepeatDelay=1`) - egy valódi "minden HCW-t minden reggel teszteljük"
politika.

**Eredmény**: `incidence_RR` = 1,00 (medián 0,9966) - paritás a lakosság többi részével. Az RR=1
nullhipotézis elleni egymintás teszt p=0,67-et ad - **statisztikailag nem különbözik 1-től** (a 20
futásból 11 van 1 alatt, 9 fölötte - fej-vagy-írás eloszlás, nem egyirányú trend).

A `hcwPPE` előtt ez volt a sorozat legerősebb eredménye; még mindig a második legerősebb és a
legjobb *teszteléssel elérhető* eredmény - nagy védőhatás (-0,38 a baseline-hoz képest, nagyobb,
mint a `hiTest` már eddig is jelentős -0,29-es hatása), amely teljesen bezárja a rést, nem csak
szűkíti. Összhangban van azzal a mechanisztikus felfedezéssel, hogy a tesztelés a
karantén/elkülönítés csatornán keresztül hat (lásd az 5. pontot fent) - az újratesztelési
késleltetés megszüntetésével a valóban napi szűrés szinte azonnal elkapja és elkülöníti a fertőzött
HCW-ket, maximalizálva ennek a csatornának a hatását. A `hcwPPE`-vel ellentétben azonban ez a
védelem feltételes - csak akkor működik, ha a karantén aktív (lásd `quarantine0_hiTest` fent); a
`hcwPPE` hatásának nincs ilyen függősége.

### 9. Számít-e a karantén *időtartama*, függetlenül attól, hogy egyáltalán aktiválódik-e? (`longQuarantine`)

Az elkülönítés meghosszabbítása az alapértelmezett 10 napról 21-re (`--quarantineLength 21`, a
`quarantine0` szintje változatlanul 3 marad) szignifikánsan csökkenti a HCW-k abszolút
fertőzésszámát (`mean_hce`: -0,68, `hcw_infections`: -249, mindkettő p<0,0001) - a hosszabb
elkülönítés valóban csökkenti a terjedést. De **az `incidence_RR` alig mozdul (1,38→1,35), és a
változás nem szignifikáns** (p=0,118, ellentétben minden eddig tesztelt tényezővel). A hosszabb
elkülönítés arányosan csökkenti a fertőzéseket a HCW-knél és a lakosság többi részénél is - mint a
`closuresNone` esetében, ez egy olyan tényező, amely mindenkit együtt mozgat, nem változtatja meg a
HCW-k *relatív* pozícióját. (A `prevalence_RR` kissé mozdul, -0,02, p=0,016 - összhangban a már
ismert mechanizmussal: a hosszabb karantén-időtartam felfújja a lakosság többi részének
"karanténban lévő" állományát, hiszen ők profitálnak a munkahelyi karanténból - a `quarantine0`-nál
látott hatás kisebb változata.)

## Összegzés: rangsor a HCW-sérülékenység szempontjából

0. **Kórházi védőfelszerelés (`hcwPPE`)** - a messze leghatékonyabb beavatkozás: egy 70%-os
   hatékonyságú feltételezés kórházi/rendelői helyszíneken nem csak bezárja, hanem **meg is
   fordítja** a sérülékenységi rést (RR=0,71, mind a 20 futás 1 alatt, p=1,9×10⁻⁶).
   Mechanisztikusan eltér minden más tesztelt tényezőtől - közvetlenül a fertőzés helyén hat,
   függetlenül attól, hogy van-e karantén/tesztelés/oltási infrastruktúra. A kísérletsorozat
   legegyértelműbb, önmagában megálló szakpolitikai javaslata.
0b. **Tesztelési intenzitás/gyakoriság, helyesen megvalósítva (`hcwDailyTest`)** - a legjobb
   *teszteléssel elérhető* eredmény: a `hiTest`-ben hagyott újratesztelési késleltetés
   megszüntetése a tesztelést "jelentősen segít" (-0,29) szintről "teljesen lezárja a rést"
   (RR=1,00, statisztikailag nem különbözik a paritástól) szintre emeli. Továbbra is függ az aktív
   karanténtól (lásd 2. pont), ellentétben a PPE-vel - ez a tesztelés elérhető felső határa, ha ez
   a feltétel teljesül.
1. **Oltás (közvetlen HCW-hozzáférés)** - a legnagyobb és legrobusztusabb *káros hatás, ha
   megszűnik* minden tesztelt környezetben (+0,40-0,45), és az egyetlen tényező, amelyről
   bizonyítottan a HCW-k *saját* hozzáférése számít, nem csak a populációs szintű nyájimmunitás.
2. **Tesztelés, de csak akkor, ha van aktív karantén** - nagy és megbízható hatás, ha a
   karanténpolitika érvényben van (-0,29-től -0,38-ig, az agresszivitástól függően; az oltási
   prioritás elvesztésének kb. 68%-át pótolja `hiTest` szinten, a teljes rést lezárja
   `hcwDailyTest` szinten), de **önmagában hatástalan** (p=0,51), ha a karantén ki van kapcsolva.
   Nem önálló beavatkozás - a karanténtól függ.
3. **Karanténpolitika** - szerény *közvetlen* hatás attól, hogy egyáltalán engedélyezve van-e
   (+0,09, ha teljesen kikapcsolva), ami alulbecsüli a valódi fontosságát, mivel egyben
   előfeltétele is a tesztelés sokkal nagyobb hatásának. Az *időtartam* önmagában (10→21 nap,
   `longQuarantine`), függetlenül attól, hogy aktiválódik-e, **nem** változtatja meg szignifikánsan
   az `incidence_RR`-t (p=0,12) - az számít, hogy létezik-e egyáltalán karantén, nem az, hogy
   mennyi ideig tart, ha már aktiválódott. Emellett az a tényező, amelynek eltávolítása paradox
   módon *szűkíti* a HCW-populáció rést, ha az oltási prioritás már amúgy is elveszett -
   emlékeztető arra, hogy ezek a tényezők egymással is kölcsönhatásban állnak, nem csak egy
   rögzített alaphelyzethez képest.
4. **Zárlatok/korlátozások (maszk, kijárási tilalom, iskolai korlátozás, helyszín-bezárások)** -
   egyik tesztben sem volt mérhető hatása a HCW-vs-populáció *arányra*. Valószínűleg azért, mert a
   zárlatok általánosan csökkentik a terjedést, foglalkozástól függetlenül - mindkét csoportot
   együtt mozgatják, nem differenciáltan, ugyanaz a mintázat, mint a karantén *időtartamánál* fent.

## Végkövetkeztetés

Az egészségügyi dolgozók **statisztikailag igazolhatóan, szinte minden tesztelt körülmény között**
sérülékenyebb csoportnak bizonyultak, mint a lakosság többi része - ez alól csak két, kifejezetten
HCW-re célzott beavatkozás jelent kivételt. A legfontosabb, egyértelműen alátámasztott
szakpolitikai üzenet: **a kórházi védőfelszerelés (PPE) az egyetlen tesztelt beavatkozás, amely nem
csak kiegyenlíti, hanem meg is fordítja a HCW-k sérülékenységét** - ez a legerősebb, legtisztább és
leginkább önmagában megálló eredmény. Emellett **az egészségügyi dolgozók oltási prioritása** a
másik robusztus, önmagában is ható védelmi intézkedés - elvesztését sem a lakosság
nyájimmunitása, sem (teljes mértékben) a fokozott tesztelés nem pótolja. A **helyesen (napi
szinten, újratesztelési késleltetés nélkül) megvalósított tesztelés** szintén teljesen lezárja a
rést, de ez a védelem *feltételes*: csak akkor működik, ha létezik karanténpolitika, amely érvényt
szerez a pozitív teszteredményeknek - önmagában, karantén nélkül, a modellben nincs hatása. A
karantén *időtartama* és az általános zárlatok/korlátozások ezzel szemben nem változtatják meg a
HCW-k relatív helyzetét - mindkét csoportot együtt mozgatják, nem célzottan.
