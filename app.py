import streamlit as st
import pandas as pd
import math
import os
import csv
import re
import uuid
from datetime import datetime

DATA_FILE = "farm_data.csv"
ACCOUNTS_FILE = "accounts.csv"
MESSAGES_FILE = "messages.csv"
POSTS_FILE = "posts.csv"
LIKES_FILE = "likes.csv"
PROFILE_PIC_DIR = "profile_pics"
POST_IMAGE_DIR = "post_images"

os.makedirs(PROFILE_PIC_DIR, exist_ok=True)
os.makedirs(POST_IMAGE_DIR, exist_ok=True)


# ============================================================
# DIVERSITY CALCULATIONS
# ============================================================
def calculate_diversity(crop_names, crop_areas):
    """
    Returns Shannon-entropy-based diversity metrics for a farm.

    - variety_score: 0-100%, how evenly land is spread across crops
      (100% = perfectly even spread across all crops grown)
    - effective_crops: "effective number of crops" (Hill number, exp(entropy)).
      Easier to read than a raw entropy value — e.g. 2.3 means the farm's
      land spread behaves like ~2.3 equally-sized crops, even if there are
      technically 5 crops (a few of them tiny).
    """
    valid_crops = [(n, a) for n, a in zip(crop_names, crop_areas) if a > 0]
    total_area = sum(a for _, a in valid_crops)

    if total_area == 0 or not valid_crops:
        return None

    proportions = [a / total_area for _, a in valid_crops]
    entropy = -sum(p * math.log(p) for p in proportions if p > 0)
    num_valid_crops = len(valid_crops)

    effective_crops = math.exp(entropy) if entropy > 0 else 1.0

    max_entropy = math.log(num_valid_crops) if num_valid_crops > 1 else None
    if max_entropy and max_entropy > 0:
        variety_score = (entropy / max_entropy) * 100
    else:
        # Only one crop grown at all -> no diversity by definition
        variety_score = 0.0

    # Per-crop share of the diversity "risk" - useful for the resilience section
    crop_breakdown = []
    for (n, a), p in zip(valid_crops, proportions):
        remaining_pct = ((total_area - a) / total_area) * 100
        crop_breakdown.append({
            "name": n,
            "area": a,
            "share_pct": p * 100,
            "remaining_if_lost_pct": remaining_pct,
        })

    return {
        "variety_score": variety_score,
        "effective_crops": effective_crops,
        "num_crops": num_valid_crops,
        "valid_crops": valid_crops,
        "total_area": total_area,
        "crop_breakdown": crop_breakdown,
    }


def verdict_for_score(score):
    if score >= 70:
        return "🌱 Great variety!", "Your land is spread evenly across your crops — a shock to one crop won't sink the whole farm."
    elif score >= 40:
        return "🙂 Okay variety", "Reasonable spread, but one or two crops dominate your land."
    else:
        return "⚠️ Low variety", "Most of your land is riding on very few crops — worth considering a bit more spread."


# ============================================================
# STORAGE: FARM DATA (now editable, not append-only)
# ============================================================
def load_all_farms():
    if not os.path.isfile(DATA_FILE):
        return pd.DataFrame(columns=["farmer_name", "barangay", "crop", "area"])
    return pd.read_csv(DATA_FILE)


def save_farm_data(farmer_name, barangay, valid_crops):
    """Overwrites this farmer's previous entries instead of endlessly appending duplicates."""
    all_farms = load_all_farms()
    all_farms = all_farms[all_farms["farmer_name"] != farmer_name]
    new_rows = pd.DataFrame(
        [{"farmer_name": farmer_name, "barangay": barangay, "crop": n, "area": a} for n, a in valid_crops]
    )
    updated = pd.concat([all_farms, new_rows], ignore_index=True)
    updated.to_csv(DATA_FILE, index=False)


def load_farmer_crops(farmer_name):
    all_farms = load_all_farms()
    mine = all_farms[all_farms["farmer_name"] == farmer_name]
    if mine.empty:
        return []
    return list(zip(mine["crop"], mine["area"]))


def delete_farm_data(farmer_name):
    all_farms = load_all_farms()
    all_farms = all_farms[all_farms["farmer_name"] != farmer_name]
    all_farms.to_csv(DATA_FILE, index=False)


# ============================================================
# STORAGE: ACCOUNTS
# ============================================================
def register_account(name, role, barangay):
    existing = load_accounts()
    if not existing.empty and name in existing["name"].values:
        return
    file_exists = os.path.isfile(ACCOUNTS_FILE)
    with open(ACCOUNTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["name", "role", "barangay"])
        writer.writerow([name, role, barangay])


def load_accounts():
    if not os.path.isfile(ACCOUNTS_FILE):
        return pd.DataFrame(columns=["name", "role", "barangay"])
    return pd.read_csv(ACCOUNTS_FILE)


# ============================================================
# STORAGE: MESSAGES
# ============================================================
def send_message(from_name, to_name, message):
    file_exists = os.path.isfile(MESSAGES_FILE)
    with open(MESSAGES_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["from_name", "to_name", "message", "timestamp"])
        writer.writerow([from_name, to_name, message, datetime.now().strftime("%Y-%m-%d %H:%M")])


def load_conversation(user_a, user_b):
    if not os.path.isfile(MESSAGES_FILE):
        return pd.DataFrame(columns=["from_name", "to_name", "message", "timestamp"])
    df = pd.read_csv(MESSAGES_FILE)
    return df[
        ((df["from_name"] == user_a) & (df["to_name"] == user_b)) |
        ((df["from_name"] == user_b) & (df["to_name"] == user_a))
    ]


# ============================================================
# STORAGE: POSTS (now persisted to disk, not just session state)
# ============================================================
def save_post(author, barangay, caption, image_path):
    post_id = uuid.uuid4().hex[:10]
    file_exists = os.path.isfile(POSTS_FILE)
    with open(POSTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["post_id", "author", "barangay", "caption", "image_path", "timestamp"])
        writer.writerow([post_id, author, barangay, caption, image_path or "",
                          datetime.now().strftime("%Y-%m-%d %H:%M")])
    return post_id


def load_posts(author_filter=None):
    if not os.path.isfile(POSTS_FILE):
        df = pd.DataFrame(columns=["post_id", "author", "barangay", "caption", "image_path", "timestamp"])
    else:
        df = pd.read_csv(POSTS_FILE, dtype={"image_path": str}).fillna("")
    if author_filter:
        df = df[df["author"] == author_filter]
    return df.sort_values("timestamp", ascending=False)


def seed_posts_if_empty():
    if not os.path.isfile(POSTS_FILE):
        save_post("Aling Nena's Farm", "Barangay San Antonio",
                   "Harvested kamote and okra this week! Still hoping to find talong seedlings to trade 🌱", "")
        save_post("Kuya Bert's Farm", "Barangay Landayan",
                   "Looking for: mungbean seeds. Anyone have extra?", "")


# ============================================================
# STORAGE: LIKES
# ============================================================
def toggle_like(post_id, user):
    likes = load_likes()
    already_liked = not likes[(likes["post_id"] == post_id) & (likes["liker"] == user)].empty
    if already_liked:
        likes = likes[~((likes["post_id"] == post_id) & (likes["liker"] == user))]
        likes.to_csv(LIKES_FILE, index=False)
    else:
        file_exists = os.path.isfile(LIKES_FILE)
        with open(LIKES_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["post_id", "liker"])
            writer.writerow([post_id, user])


def load_likes():
    if not os.path.isfile(LIKES_FILE):
        return pd.DataFrame(columns=["post_id", "liker"])
    return pd.read_csv(LIKES_FILE)


def like_count(likes_df, post_id):
    return len(likes_df[likes_df["post_id"] == post_id])


def user_liked(likes_df, post_id, user):
    return not likes_df[(likes_df["post_id"] == post_id) & (likes_df["liker"] == user)].empty


# ============================================================
# PROFILE PICTURES
# ============================================================
def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def save_profile_picture(name, uploaded_file):
    path = os.path.join(PROFILE_PIC_DIR, f"{safe_filename(name)}.png")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def get_profile_picture_path(name):
    path = os.path.join(PROFILE_PIC_DIR, f"{safe_filename(name)}.png")
    return path if os.path.isfile(path) else None


def save_post_image(uploaded_file):
    filename = f"{uuid.uuid4().hex[:10]}.png"
    path = os.path.join(POST_IMAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ============================================================
# APP SETUP
# ============================================================
seed_posts_if_empty()

defaults = {
    "role": None,
    "account_created": False,
    "farmer_name": "",
    "barangay": "",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.set_page_config(page_title="AgriEntropy", page_icon="🌾", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #F7FAF6; }
div.stButton > button { background-color: #6B9971; color: white; border-radius: 10px; border: none; }
div.stButton > button:hover { background-color: #55805D; color: white; }
div.stTextInput > div > div > input { border-radius: 20px; padding-left: 14px; }

.brand-header {
    font-size: 34px;
    font-weight: 800;
    color: #3F6D48;
    letter-spacing: -0.5px;
    margin-bottom: 0px;
}
.brand-sub {
    color: #7A9483;
    font-size: 13px;
    margin-top: -6px;
    margin-bottom: 14px;
}

.post-card {
    background: white;
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #EAEFEA;
}
.post-author { font-weight: 700; font-size: 15px; color: #2E4A34; }
.post-location { font-size: 12px; color: #8A9E90; margin-top: -4px; }
.post-time { font-size: 11px; color: #B0BDB4; }

.avatar-circle {
    width: 42px; height: 42px; border-radius: 50%;
    background: #6B9971; color: white; display: flex;
    align-items: center; justify-content: center;
    font-weight: 700; font-size: 16px;
}

.diversity-box {
    background: linear-gradient(135deg, #6B9971, #4C7A57);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    color: white;
    margin-bottom: 12px;
}
.diversity-box .big-number { font-size: 52px; font-weight: 800; margin: 0; }
.diversity-box .label { font-size: 14px; opacity: 0.9; margin-top: -6px; }
.diversity-box .verdict { font-size: 18px; font-weight: 700; margin-top: 10px; }
.diversity-box .verdict-sub { font-size: 13px; opacity: 0.85; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


def log_out():
    st.session_state.role = None
    st.session_state.account_created = False
    st.session_state.farmer_name = ""
    st.session_state.barangay = ""


def brand_header():
    st.markdown('<p class="brand-header">🌾 AgriEntropy</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">San Pedro farmer network</p>', unsafe_allow_html=True)


def render_avatar(name, size=42):
    pic_path = get_profile_picture_path(name)
    if pic_path:
        st.image(pic_path, width=size)
    else:
        initial = name.strip()[0].upper() if name.strip() else "?"
        st.markdown(f'<div class="avatar-circle" style="width:{size}px;height:{size}px;">{initial}</div>',
                    unsafe_allow_html=True)


def render_feed(current_user, editable_composer=True, barangay=""):
    if editable_composer:
        with st.expander("➕ Add a post"):
            new_caption = st.text_area("What's happening on your farm?", key="new_post_text")
            new_image = st.file_uploader("Add a photo (optional)", type=["png", "jpg", "jpeg"], key="new_post_image")
            if st.button("Post", key="submit_post"):
                if new_caption.strip():
                    image_path = save_post_image(new_image) if new_image is not None else ""
                    save_post(current_user, barangay, new_caption.strip(), image_path)
                    st.rerun()
                else:
                    st.warning("Write something before posting.")

    posts = load_posts()
    likes_df = load_likes()

    for _, post in posts.iterrows():
        with st.container():
            st.markdown('<div class="post-card">', unsafe_allow_html=True)
            col_pic, col_text = st.columns([1, 6])
            with col_pic:
                render_avatar(post["author"])
            with col_text:
                st.markdown(f'<div class="post-author">{post["author"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="post-location">{post["barangay"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="post-time">{post["timestamp"]}</div>', unsafe_allow_html=True)

            st.write(post["caption"])
            if post["image_path"] and os.path.isfile(post["image_path"]):
                st.image(post["image_path"], use_container_width=True)

            n_likes = like_count(likes_df, post["post_id"])
            liked = user_liked(likes_df, post["post_id"], current_user)
            like_label = f"{'❤️' if liked else '🤍'} {n_likes}"
            if st.button(like_label, key=f"like_{post['post_id']}"):
                toggle_like(post["post_id"], current_user)
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)


def render_messages_section(current_user):
    st.subheader("💬 Messages")
    accounts = load_accounts()
    other_accounts = accounts[accounts["name"] != current_user]["name"].tolist()

    if not other_accounts:
        st.write("No other accounts to message yet.")
        return

    chat_with = st.selectbox("Chat with", other_accounts)
    convo = load_conversation(current_user, chat_with)

    with st.container(border=True, height=250):
        for _, row in convo.iterrows():
            if row["from_name"] == current_user:
                st.markdown(
                    f"<div style='text-align:right; background:#DCF2D1; padding:6px 12px; "
                    f"border-radius:12px; margin:4px 0; display:inline-block; float:right; clear:both;'>{row['message']}</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='text-align:left; background:#EEE; padding:6px 12px; "
                    f"border-radius:12px; margin:4px 0; display:inline-block; float:left; clear:both;'>{row['message']}</div>",
                    unsafe_allow_html=True)

    new_msg = st.text_input("Type a message", key=f"chat_input_{chat_with}")
    if st.button("Send", key=f"send_{chat_with}"):
        if new_msg.strip():
            send_message(current_user, chat_with, new_msg.strip())
            st.rerun()


def render_profile_picture_section(current_user, upload_key):
    pic_path = get_profile_picture_path(current_user)
    col_pic, col_info = st.columns([1, 2])
    with col_pic:
        if pic_path:
            st.image(pic_path, width=120)
        else:
            st.markdown("🧑 *(no picture yet)*")
        uploaded_pic = st.file_uploader("Add profile picture", type=["png", "jpg", "jpeg"], key=upload_key)
        if uploaded_pic is not None:
            save_profile_picture(current_user, uploaded_pic)
            st.success("Profile picture updated!")
            st.rerun()
    return col_info


def render_diversity_section(farmer_name, barangay):
    st.subheader("🌱 My Farm")

    saved_crops = load_farmer_crops(farmer_name)
    default_count = max(len(saved_crops), 1)

    num_crops = st.number_input("How many different crops do you grow?", min_value=1, max_value=15,
                                 value=default_count, step=1)

    crop_names, crop_areas = [], []
    for i in range(int(num_crops)):
        default_name = saved_crops[i][0] if i < len(saved_crops) else ""
        default_area = float(saved_crops[i][1]) if i < len(saved_crops) else 0.0
        c1, c2 = st.columns(2)
        with c1:
            name_c = st.text_input(f"Crop {i+1} name", value=default_name, key=f"name_{i}")
        with c2:
            area = st.number_input("Land used (sq. m)", min_value=0.0, value=default_area, key=f"area_{i}")
        crop_names.append(name_c if name_c else f"Crop {i+1}")
        crop_areas.append(area)

    col_calc, col_reset = st.columns([2, 1])
    with col_calc:
        calc_clicked = st.button("Calculate my farm's diversity")
    with col_reset:
        if st.button("🗑️ Clear saved data"):
            delete_farm_data(farmer_name)
            st.success("Cleared. Refreshing...")
            st.rerun()

    if calc_clicked:
        result = calculate_diversity(crop_names, crop_areas)
        if result is None:
            st.warning("Please enter some land area for at least one crop.")
        else:
            st.session_state.last_result = result

    if "last_result" in st.session_state:
        result = st.session_state.last_result
        score = result["variety_score"]
        verdict, verdict_sub = verdict_for_score(score)

        st.markdown(f"""
        <div class="diversity-box">
            <p class="label">FARM HEALTH SCORE</p>
            <p class="big-number">{score:.0f}%</p>
            <p class="verdict">{verdict}</p>
            <p class="verdict-sub">{verdict_sub}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### {farmer_name}")
        st.caption(f"📍 {barangay}")

        m1, m2 = st.columns(2)
        m1.metric("Total farm area", f"{result['total_area']:.0f} m²")
        m2.metric("Effective number of crops", f"{result['effective_crops']:.1f}",
                   help="Accounts for how evenly land is split. E.g. 5 crops where one takes up "
                        "90% of the land behaves like ~1.3 'effective' crops, not 5.")

        st.subheader("Crops on this farm")
        table_data = [{"Crop": c["name"], "Land area (sq.m)": c["area"], "Share of farm": f"{c['share_pct']:.1f}%"}
                      for c in result["crop_breakdown"]]
        st.table(table_data)

        st.subheader("🛡️ What if one crop fails?")
        for c in result["crop_breakdown"]:
            st.write(f"If **{c['name']}** fails, you'd keep **{c['remaining_if_lost_pct']:.0f}%** of production.")

        if st.button("💾 Save this data"):
            save_farm_data(farmer_name, barangay, result["valid_crops"])
            st.success("Saved! This will pre-fill next time you open My Farm.")


def render_search(placeholder_text, key):
    st.subheader("Search")
    query = st.text_input("", placeholder=placeholder_text, key=key)
    all_farms = load_all_farms()
    if all_farms.empty:
        st.write("No registered farms yet.")
        return
    unique_farms = all_farms[["farmer_name", "barangay"]].drop_duplicates()
    if query:
        unique_farms = unique_farms[
            unique_farms["farmer_name"].str.contains(query, case=False, na=False) |
            unique_farms["barangay"].str.contains(query, case=False, na=False)
        ]
    if unique_farms.empty:
        st.write("No matches found.")
    for _, row in unique_farms.iterrows():
        with st.container(border=True):
            col_pic, col_text = st.columns([1, 6])
            with col_pic:
                render_avatar(row["farmer_name"], size=36)
            with col_text:
                st.markdown(f"**{row['farmer_name']}** — {row['barangay']}")


# ============================================================
# SCREEN 1: ROLE SELECTION
# ============================================================
if st.session_state.role is None:
    brand_header()
    st.write("Are you a...")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍🌾 Farmer", use_container_width=True):
            st.session_state.role = "farmer"
            st.rerun()
    with col2:
        if st.button("🏛️ Government Office", use_container_width=True):
            st.session_state.role = "gov"
            st.rerun()

# ============================================================
# SCREEN 2: ACCOUNT CREATION
# ============================================================
elif not st.session_state.account_created:
    role_label = "Farmer" if st.session_state.role == "farmer" else "Government"
    st.title(f"Create your {role_label} account")

    name = st.text_input("Your name")
    barangay = ""
    if st.session_state.role == "farmer":
        barangay = st.text_input("Your barangay")

    if st.button("Create account"):
        if not name.strip():
            st.warning("Please enter your name.")
        elif st.session_state.role == "farmer" and not barangay.strip():
            st.warning("Please enter your barangay.")
        else:
            st.session_state.farmer_name = name.strip()
            st.session_state.barangay = barangay.strip()
            register_account(name.strip(), st.session_state.role, barangay.strip())
            st.session_state.account_created = True
            st.rerun()

    if st.button("← Back"):
        st.session_state.role = None
        st.rerun()

# ============================================================
# FARMER APP
# ============================================================
elif st.session_state.role == "farmer":
    st.sidebar.write(f"Signed in as **{st.session_state.farmer_name}**")
    st.sidebar.button("← Log out", on_click=log_out)

    tab_home, tab_search, tab_data, tab_profile = st.tabs(["🏠 Home", "🔍 Search", "🌱 My Farm", "👤 Profile"])

    with tab_home:
        brand_header()
        st.subheader("Nearby farms")
        render_feed(st.session_state.farmer_name, editable_composer=True, barangay=st.session_state.barangay)

    with tab_search:
        render_search("🔍 Search farms, people, or barangays", "search_farmer")

    with tab_data:
        render_diversity_section(st.session_state.farmer_name, st.session_state.barangay)

    with tab_profile:
        st.subheader("👤 Profile")
        col_info = render_profile_picture_section(st.session_state.farmer_name, "pic_upload_farmer")
        with col_info:
            st.write(f"**Name:** {st.session_state.farmer_name}")
            st.write(f"**Barangay:** {st.session_state.barangay}")

        st.divider()
        st.subheader("📸 My posts")
        my_posts = load_posts(author_filter=st.session_state.farmer_name)
        if my_posts.empty:
            st.caption("You haven't posted anything yet.")
        else:
            cols = st.columns(3)
            for i, (_, post) in enumerate(my_posts.iterrows()):
                with cols[i % 3]:
                    if post["image_path"] and os.path.isfile(post["image_path"]):
                        st.image(post["image_path"], use_container_width=True)
                    else:
                        st.markdown(f"*{post['caption'][:60]}*")

        st.divider()
        render_messages_section(st.session_state.farmer_name)

        st.divider()
        st.subheader("🔒 Data & privacy")
        all_farms = load_all_farms()
        if not all_farms.empty:
            my_data = all_farms[all_farms["farmer_name"] == st.session_state.farmer_name]
            if not my_data.empty:
                csv_bytes = my_data.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Request a copy of my data", data=csv_bytes,
                                    file_name="my_farm_data.csv", mime="text/csv")
            else:
                st.write("No saved data yet.")
        else:
            st.write("No saved data yet.")

# ============================================================
# GOVERNMENT APP
# ============================================================
elif st.session_state.role == "gov":
    st.sidebar.write(f"Signed in as **{st.session_state.farmer_name}**")
    st.sidebar.button("← Log out", on_click=log_out)

    tab_home, tab_search, tab_profile = st.tabs(["🏠 Home", "🔍 Search", "👤 Profile"])
    all_farms = load_all_farms()

    with tab_home:
        brand_header()
        st.subheader("San Pedro City — Overview")

        if all_farms.empty:
            st.info("No farm data submitted yet. Once farmers save their data, it will appear here.")
        else:
            scores = []
            for farm_name, group in all_farms.groupby("farmer_name"):
                result = calculate_diversity(list(group["crop"]), list(group["area"]))
                if result:
                    scores.append({
                        "Farm": farm_name,
                        "Barangay": group["barangay"].iloc[0],
                        "Score": result["variety_score"],
                        "Effective crops": result["effective_crops"],
                    })

            if scores:
                avg_score = sum(s["Score"] for s in scores) / len(scores)
                avg_effective = sum(s["Effective crops"] for s in scores) / len(scores)
                m1, m2 = st.columns(2)
                m1.metric("City-wide average Farm Health Score", f"{avg_score:.0f}%")
                m2.metric("Average effective crops per farm", f"{avg_effective:.1f}")

                st.subheader("📋 Registered farms")
                st.table([{"Farm": s["Farm"], "Barangay": s["Barangay"],
                          "Farm Health Score": f"{s['Score']:.0f}%",
                          "Effective crops": f"{s['Effective crops']:.1f}"} for s in scores])

                low_variety = [s["Farm"] for s in scores if s["Score"] < 40]
                if low_variety:
                    st.warning(f"⚠️ Farms that may need support: {', '.join(low_variety)}")

        st.divider()
        st.subheader("📢 Community feed")
        render_feed(st.session_state.farmer_name, editable_composer=False)

    with tab_search:
        render_search("🔍 Search farmers or barangays", "search_gov")

    with tab_profile:
        st.subheader("👤 Profile")
        col_info = render_profile_picture_section(st.session_state.farmer_name, "pic_upload_gov")
        with col_info:
            st.write(f"**Officer:** {st.session_state.farmer_name}")
            st.write("San Pedro City Hall")

        st.divider()
        render_messages_section(st.session_state.farmer_name)

        st.divider()
        st.subheader("📄 Reports")
        if not all_farms.empty:
            csv_bytes = all_farms.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download full city report", data=csv_bytes,
                                file_name="san_pedro_farm_report.csv", mime="text/csv")
        else:
            st.write("No data to report yet.")
