# =====================================================================
# Türkçe Haber Duygu Analizi - Streamlit Arayüzü
# Çalıştırma:  streamlit run app.py
# Gereken dosya: aynı klasörde `duygu_modeli.joblib` (Colab'da Hücre 4 üretir)
# =====================================================================

import os
import re
import time
from urllib.parse import urljoin, urlparse

import feedparser
import joblib
import numpy as np
import pandas as pd
import requests
import snowballstemmer
import streamlit as st
import trafilatura
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Sayfa ayarları
# ---------------------------------------------------------------------
st.set_page_config(page_title="Haber Duygu Analizi", page_icon="📰", layout="wide")

MODEL_DOSYASI = "duygu_modeli.joblib"
ADLAR = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
SUTUN_ADLARI = {0: "negatif", 1: "notr", 2: "pozitif"}  # CSV/sütun adları için ASCII
SIRALI_ETIKETLER = ["Negatif", "Nötr", "Pozitif"]
RENKLER = ["#d62728", "#7f7f7f", "#2ca02c"]

MIN_METIN_UZUNLUGU = 300   # karakter; daha kısa metinler geçersiz sayılır
BEKLEME_SN = 1.0           # istekler arası bekleme
BASLIKLAR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# Gerçek haber linki: yolun sonunda "-" + en az 6 haneli sayısal ID (isteğe bağlı .htm/.html ve "/")
# Kategori sayfaları, döviz/borsa ekranları ve KVKK metinleri bu desene uymadığı için elenir.
HABER_ID_DESENI = re.compile(r"-\d{6,}(?:\.html?)?/?$")


# ---------------------------------------------------------------------
# Türkçe ön işleme (Colab'daki temizle_metin ile birebir aynı olmalı!)
# ---------------------------------------------------------------------
TR_STOPWORDS = set("""
acaba ama ancak artık aslında az bana bazen bazı belki ben beni benim beri bile bir birçok biri birkaç birşey biz bizi bizim
böyle böylece bu buna bunda bundan bunu bunun burada bütün çok çünkü da daha dahi de defa diye diğer en gibi göre hem hep hepsi
her herhangi herkes hiçbir için ile ise kadar kendi kendine kez ki kim kime kimi kimse mı mi mu mü nasıl ne neden nerede nereye
niçin niye o olarak oldu olduğu olan olsa olur olmuş on ona ondan onlar onlara onları onun orada öyle pek sadece sanki şey şu
şuna şunu siz sizi sizin sonra tüm ve veya ya yani yine yoksa zaten üzere ise
""".split())

STEMMER = snowballstemmer.stemmer("turkish")


def turkce_kucuk_harf(metin):
    """Türkçe'ye uygun küçük harf dönüşümü: İ->i, I->ı."""
    return metin.replace("İ", "i").replace("I", "ı").lower()


def temizle_metin(metin):
    """HTML/URL/sayı/noktalama temizliği + stopword + Snowball kök bulma."""
    if not isinstance(metin, str):
        return ""
    metin = re.sub(r"<[^>]+>", " ", metin)
    metin = re.sub(r"http\S+|www\.\S+", " ", metin)
    metin = turkce_kucuk_harf(metin)
    metin = re.sub(r"\d+", " ", metin)
    metin = re.sub(r"[^a-zçğıöşüâîû\s]", " ", metin)
    kelimeler = [k for k in metin.split() if k not in TR_STOPWORDS and len(k) > 1]
    return " ".join(STEMMER.stemWords(kelimeler))


# ---------------------------------------------------------------------
# Model yükleme (bir kez yüklenir, oturumlar arası paylaşılır)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor...")
def modeli_yukle(yol):
    paket = joblib.load(yol)
    return paket["vektorlestirici"], paket["yorum_modeli"], paket["olasilik_modeli"]


if not os.path.exists(MODEL_DOSYASI):
    st.error(
        f"`{MODEL_DOSYASI}` bulunamadı. Colab'da Hücre 4'ü çalıştırıp oluşan dosyayı "
        "bu uygulamayla aynı klasöre koyun."
    )
    st.stop()

vektorlestirici, yorum_modeli, olasilik_modeli = modeli_yukle(MODEL_DOSYASI)


# ---------------------------------------------------------------------
# Veri toplama
# ---------------------------------------------------------------------
# ESKİ HALİ:
# HABER_ID_DESENI = re.compile(r"-\d{6,}(?:\.html?)?/?$")

# YENİ HALİ: 6+ haneli ulusal ID'ler VEYA kısa ID'ler (-123) VEYA en az 3 tireli haber başlığı slug'ları
HABER_ID_DESENI = re.compile(r"(?:[-/]\d{1,10}(?:\.html?)?/?$|(?:[a-zçğıöşü0-9]+-){3,}[a-zçğıöşü0-9]+/?$)")

YASAKLI_DESENLER = (
    "kvkk", "kisisel-verilerin", "kullanim-kosullari", "gizlilik-politikasi",
    "enerji-politikasi", "namaz-vakitleri", "canli-borsa", "hesaplama-araclari",
    "/haberleri/", "/doviz/", "/kunye", "/iletisim", "/hakkimizda", "/etiket/", "/tag/"
)

def haber_linki_mi(url):
    """Ulusal ve yerel haber sitelerindeki tekil haber linklerini kabul eder, kurumsal/kategori sayfalarını eler."""
    try:
        url_kucuk = url.lower().strip()
        if any(y in url_kucuk for y in YASAKLI_DESENLER):
            return False
        yol = urlparse(url_kucuk).path.strip("/")
        if len(yol) < 12:
            return False
    except Exception:
        return False
    return bool(HABER_ID_DESENI.search(yol))


def rss_linklerini_al(url):
    """RSS'ten (link, başlık, tarih) listesi döndürür; yalnızca haber linkleri."""
    try:
        akis = feedparser.parse(url, request_headers=BASLIKLAR)
    except Exception:
        return []
    sonuc, gorulen = [], set()
    for e in akis.entries:
        link = e.get("link")
        if link and link not in gorulen and haber_linki_mi(link):
            gorulen.add(link)
            sonuc.append((link, e.get("title", ""), e.get("published", e.get("updated", ""))))
    return sonuc


def html_linklerini_al(url):
    """RSS çalışmazsa sayfadaki haber linklerini ayıklar (aynı alan adı + haber ID deseni)."""
    try:
        yanit = requests.get(url, headers=BASLIKLAR, timeout=20)
        yanit.raise_for_status()
    except Exception:
        return []
    corba = BeautifulSoup(yanit.text, "html.parser")
    alan = urlparse(url).netloc.replace("www.", "")
    sonuc, gorulen = [], set()
    for a in corba.find_all("a", href=True):
        link = urljoin(url, a["href"]).split("#")[0].split("?")[0]
        if urlparse(link).netloc.replace("www.", "") != alan:
            continue
        if link in gorulen or not haber_linki_mi(link):
            continue
        gorulen.add(link)
        sonuc.append((link, a.get_text(strip=True), ""))
    return sonuc


def haber_cek(link, rss_baslik="", rss_tarih=""):
    """Tek haberi indirir, trafilatura ile ana metni çıkarır; yetersizse None döner."""
    yanit = requests.get(link, headers=BASLIKLAR, timeout=20)
    yanit.raise_for_status()
    html = yanit.text
    metin = trafilatura.extract(html, include_comments=False, include_tables=False,
                                favor_precision=True)
    if not metin or len(metin) < MIN_METIN_UZUNLUGU:
        return None
    meta = trafilatura.extract_metadata(html)
    baslik = (meta.title if meta and meta.title else rss_baslik) or ""
    tarih = (meta.date if meta and meta.date else rss_tarih) or ""
    return {"tarih": tarih, "baslik": baslik.strip(), "haber_metni": metin.strip(), "url": link}


def haberleri_topla(kaynaklar, adet, ilerleme, durum):
    """kaynaklar: {ad: url}. Her kaynaktan en fazla `adet` geçerli haber toplar."""
    kayitlar, uyarilar = [], []
    toplam_hedef = adet * len(kaynaklar)
    biten = 0
    for ad, url in kaynaklar.items():
        durum.write(f"**{ad}** için bağlantılar toplanıyor...")
        adaylar = rss_linklerini_al(url) or html_linklerini_al(url)
        if not adaylar:
            uyarilar.append(f"{ad}: Haber ID desenine uyan bağlantı bulunamadı (link veya site yapısını kontrol edin).")
            biten += adet
            ilerleme.progress(min(biten / toplam_hedef, 1.0))
            continue

        toplanan, deneme = 0, 0
        for link, baslik, tarih in adaylar:
            if toplanan >= adet or deneme >= adet * 3:  # takılmayı önlemek için deneme sınırı
                break
            deneme += 1
            try:
                kayit = haber_cek(link, baslik, tarih)
            except Exception:
                kayit = None
            if kayit:
                kayit["kaynak"] = ad
                kayitlar.append(kayit)
                toplanan += 1
                ilerleme.progress(min((biten + toplanan) / toplam_hedef, 1.0))
                durum.write(f"**{ad}**: {toplanan}/{adet} haber alındı")
            time.sleep(BEKLEME_SN)
        if toplanan < adet:
            uyarilar.append(f"{ad}: yalnızca {toplanan}/{adet} geçerli haber toplanabildi.")
        biten += adet
    ilerleme.progress(1.0)
    df = pd.DataFrame(kayitlar, columns=["kaynak", "tarih", "baslik", "haber_metni", "url"])
    return df.drop_duplicates(subset="url").reset_index(drop=True), uyarilar


# ---------------------------------------------------------------------
# Tahmin
# ---------------------------------------------------------------------
def tahmin_et(metinler):
    """Metin listesi için (etiket listesi, olasılık matrisi, sınıf listesi) döndürür."""
    X = vektorlestirici.transform([temizle_metin(m) for m in metinler])
    p = olasilik_modeli.predict_proba(X)
    siniflar = list(olasilik_modeli.classes_)
    etiketler = [ADLAR[siniflar[i]] for i in p.argmax(axis=1)]
    return etiketler, p, siniflar


def analiz_et(df):
    """Başlık bazlı ve başlık+metin birleşik tahminleri ayrı sütunlar olarak ekler."""
    sonuc = df.copy()
    baslik = df["baslik"].fillna("")
    birlesik = baslik + " " + df["haber_metni"].fillna("")
    for ek, metinler in (("baslik", baslik), ("birlesik", birlesik)):
        etiketler, p, siniflar = tahmin_et(list(metinler))
        sonuc[f"duygu_{ek}"] = etiketler
        for j, s in enumerate(siniflar):
            sonuc[f"olasilik_{SUTUN_ADLARI[s]}_{ek}"] = p[:, j].round(4)
    return sonuc


def en_belirleyici_terimler(adet=15):
    """Her sınıf için en yüksek coef_ değerli terimleri tablo olarak döndürür."""
    terimler = np.array(vektorlestirici.get_feature_names_out())
    sutunlar = {}
    for i, s in enumerate(yorum_modeli.classes_):
        idx = np.argsort(yorum_modeli.coef_[i])[::-1][:adet]
        sutunlar[ADLAR[s]] = [f"{terimler[k]}  ({yorum_modeli.coef_[i][k]:.2f})" for k in idx]
    tablo = pd.DataFrame(sutunlar, index=range(1, adet + 1))
    tablo.index.name = "Sıra"
    return tablo


# ---------------------------------------------------------------------
# Yan menü
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Ayarlar")
kaynaklar = {}
for i in range(1, 4):
    st.sidebar.subheader(f"Kaynak {i}")
    ad = st.sidebar.text_input("Site adı", value="", key=f"ad_{i}", placeholder=f"Örn. Site {i}")
    url = st.sidebar.text_input("RSS / web linki", value="", key=f"url_{i}",
                                placeholder="https://...")
    if url.strip():
        kaynaklar[ad.strip() or f"Kaynak {i}"] = url.strip()

haber_sayisi = st.sidebar.slider("Kaynak başına haber sayısı", min_value=5, max_value=50, value=15, step=1)
calistir = st.sidebar.button("Haberleri Çek ve Analiz Et", type="primary", use_container_width=True)

st.sidebar.caption("Yalnızca URL sonunda en az 6 haneli sayısal ID (`-123456`) bulunan haber bağlantıları kabul edilir.")


# ---------------------------------------------------------------------
# Çalıştırma (sonuçlar session_state'te tutulur; indirme/sekme değişimi sonucu silmez)
# ---------------------------------------------------------------------
st.title("📰 Haber Duygu Analizi")
st.caption("Medya çerçeveleme çalışması için TF-IDF + doğrusal model ile Türkçe haber duygu sınıflandırması")

if calistir:
    if not kaynaklar:
        st.sidebar.warning("En az bir kaynak linki girin.")
    else:
        ilerleme = st.progress(0.0)
        durum = st.empty()
        with st.spinner("Haberler toplanıyor, lütfen bekleyin..."):
            df_haber, uyarilar = haberleri_topla(kaynaklar, haber_sayisi, ilerleme, durum)
        durum.empty()
        ilerleme.empty()
        for u in uyarilar:
            st.warning(u)
        if df_haber.empty:
            st.error("Hiç geçerli haber toplanamadı.")
            st.session_state.pop("sonuc", None)
        else:
            with st.spinner("Duygu analizi yapılıyor..."):
                st.session_state["sonuc"] = analiz_et(df_haber)
            st.success(f"{len(df_haber)} haber analiz edildi.")

sonuc = st.session_state.get("sonuc")

sekme1, sekme2, sekme3 = st.tabs(["📊 Kaynak Karşılaştırması", "📋 Haber Detayları", "🔑 Anahtar Terimler"])

# --- Sekme 1: Kaynaklara göre duygu dağılımı ---
with sekme1:
    if sonuc is None:
        st.info("Sol menüden kaynakları girip analizi başlatın.")
    else:
        taban = st.radio("Tahmin tabanı", ["Başlık Bazlı", "Başlık + Metin Birleşik"], horizontal=True)
        sutun = "duygu_baslik" if taban == "Başlık Bazlı" else "duygu_birlesik"

        adet = pd.crosstab(sonuc["kaynak"], sonuc[sutun]).reindex(columns=SIRALI_ETIKETLER, fill_value=0)
        yuzde = (adet.div(adet.sum(axis=1), axis=0) * 100).round(1)

        st.subheader("Duygu dağılımı (haber adedi)")
        st.bar_chart(adet, color=RENKLER)

        st.subheader("Yüzdelik karşılaştırma (%)")
        st.dataframe(yuzde.style.format("{:.1f}"), use_container_width=True)

# --- Sekme 2: Haber detayları ---
with sekme2:
    if sonuc is None:
        st.info("Henüz analiz yapılmadı.")
    else:
        goster = ["kaynak", "baslik", "duygu_baslik"] + \
                 [f"olasilik_{SUTUN_ADLARI[i]}_baslik" for i in range(3)] + \
                 ["duygu_birlesik"] + \
                 [f"olasilik_{SUTUN_ADLARI[i]}_birlesik" for i in range(3)] + ["url"]
        goster = [c for c in goster if c in sonuc.columns]
        st.dataframe(
            sonuc[goster],
            use_container_width=True,
            hide_index=True,
            column_config={"url": st.column_config.LinkColumn("Link", display_text="Aç")},
        )
        st.download_button(
            "CSV olarak indir",
            data=sonuc.to_csv(index=False).encode("utf-8-sig"),  # Excel'de Türkçe karakterler bozulmasın
            file_name="analiz_sonuclari.csv",
            mime="text/csv",
        )

# --- Sekme 3: Modelin en belirleyici terimleri ---
with sekme3:
    st.subheader("Sınıf başına en belirleyici 15 terim")
    st.caption("Terimler Snowball kök hâlindedir; parantez içi model katsayısıdır (coef_).")
    st.dataframe(en_belirleyici_terimler(15), use_container_width=True)
