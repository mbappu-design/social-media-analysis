"""
Analytics Engine for Social Media Analytics Platform.
Provides data parsing, metric aggregations, sentiment and emotion analysis,
engagement scoring, and content optimization recommendations.
"""

import re
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Comprehensive Sentiment & Emotion Lexicons
AFINN_LEXICON = {
    # Positive words
    "love": 3.5, "loved": 3.2, "loving": 3.2, "lovely": 3.0, "amazing": 3.5, "awesome": 3.4,
    "excellent": 3.5, "fantastic": 3.6, "great": 3.0, "good": 2.0, "best": 3.5, "better": 2.0,
    "happy": 3.0, "happiness": 3.2, "joy": 3.4, "wonderful": 3.4, "perfect": 3.5, "brilliant": 3.3,
    "impressive": 3.0, "outstanding": 3.5, "superb": 3.4, "stunning": 3.2, "delight": 3.1,
    "thrilled": 3.3, "excited": 3.1, "exciting": 3.1, "win": 2.8, "winner": 3.0, "winning": 3.0,
    "success": 3.0, "successful": 3.0, "congratulations": 3.0, "growth": 2.2, "favorite": 2.8,
    "helpful": 2.2, "recommend": 2.5, "innovative": 2.8, "valuable": 2.5, "fire": 2.4,
    "lit": 2.2, "cool": 2.0, "nice": 2.0, "clean": 1.8, "smooth": 1.8, "fast": 1.5,
    "easy": 1.8, "fun": 2.5, "beautiful": 3.0, "inspired": 2.8, "positive": 2.4, "progress": 2.0,
    "proud": 2.8, "thank": 2.0, "thanks": 2.2, "thankful": 2.8, "appreciate": 2.6, "rocking": 2.5,

    # Negative words
    "hate": -3.5, "hated": -3.2, "terrible": -3.5, "awful": -3.4, "horrible": -3.6, "bad": -2.5,
    "worst": -3.8, "worse": -2.5, "poor": -2.2, "disappoint": -2.8, "disappointed": -3.0,
    "disappointing": -2.8, "disappointment": -3.0, "sad": -2.8, "angry": -3.2, "anger": -3.0,
    "fail": -3.0, "failed": -3.0, "failing": -2.8, "failure": -3.2, "broken": -2.5, "bug": -2.0,
    "buggy": -2.6, "glitch": -2.2, "crash": -2.8, "crashed": -3.0, "slow": -2.0, "sucks": -3.5,
    "boring": -2.2, "annoying": -2.8, "annoyed": -2.8, "frustrated": -3.0, "frustrating": -3.0,
    "scam": -3.8, "fraud": -3.8, "waste": -3.0, "useless": -3.2, "rubbish": -3.0, "trash": -3.2,
    "toxic": -3.4, "painful": -2.8, "problem": -2.0, "problems": -2.0, "issue": -1.8, "issues": -1.8,
    "lose": -2.5, "lost": -2.4, "losing": -2.5, "ugly": -2.6, "unhappy": -2.8, "negative": -2.2,
    "complaint": -2.4, "complain": -2.2, "ridiculous": -2.8, "unfair": -2.5, "disaster": -3.5
}

EMOTION_KEYWORDS = {
    "Joy": ["love", "happy", "excited", "awesome", "great", "celebrate", "win", "thrilled", "blessed", "proud", "delight", "fun", "cheers", "magic"],
    "Anticipation": ["soon", "launch", "upcoming", "excited", "wait", "ready", "tomorrow", "future", "next", "announcement", "preview", "roadmap"],
    "Surprise": ["wow", "unexpected", "unbelievable", "omg", "shocked", "mindblown", "huge", "insane", "astonishing", "incredible"],
    "Frustration": ["annoying", "frustrated", "slow", "broken", "bug", "why", "again", "fix", "stuck", "delay", "waiting", "fail"],
    "Sadness": ["sad", "miss", "sorry", "disappointed", "unfortunate", "heartbroken", "goodbye", "crying", "regret", "loss"]
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def analyze_sentiment_text(text: str) -> Dict[str, Any]:
    """Analyzes text sentiment, polarity score, subjectivity, and emotional breakdown."""
    if not text or not isinstance(text, str):
        return {
            "score": 0.0,
            "label": "Neutral",
            "compound": 0.0,
            "positive_pct": 33.3,
            "neutral_pct": 33.4,
            "negative_pct": 33.3,
            "subjectivity": 0.0,
            "emotions": {k: 0 for k in EMOTION_KEYWORDS},
            "dominant_emotion": "Neutral",
            "keywords": []
        }

    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return {
            "score": 0.0,
            "label": "Neutral",
            "compound": 0.0,
            "positive_pct": 33.3,
            "neutral_pct": 33.4,
            "negative_pct": 33.3,
            "subjectivity": 0.0,
            "emotions": {k: 0 for k in EMOTION_KEYWORDS},
            "dominant_emotion": "Neutral",
            "keywords": []
        }

    pos_score = 0.0
    neg_score = 0.0
    subjective_words = 0
    negation = False
    negation_words = {"not", "no", "never", "hardly", "barely", "scarcely", "isn't", "aren't", "wasn't", "don't", "doesn't", "didn't", "won't", "cannot", "cant"}

    emotions_count = {k: 0 for k in EMOTION_KEYWORDS}

    for idx, token in enumerate(tokens):
        # Negation detection window of 2 words
        if token in negation_words:
            negation = True
            continue

        weight = AFINN_LEXICON.get(token, 0.0)
        if weight != 0.0:
            subjective_words += 1
            if negation:
                weight = -weight * 0.75
                negation = False

            if weight > 0:
                pos_score += weight
            else:
                neg_score += abs(weight)

        # Emotion counts
        for emotion, kw_list in EMOTION_KEYWORDS.items():
            if token in kw_list:
                emotions_count[emotion] += 1

    total_weight = pos_score + neg_score
    raw_delta = pos_score - neg_score
    
    # Normalized compound between -1.0 and 1.0
    compound = raw_delta / math.sqrt(raw_delta**2 + 15.0) if raw_delta != 0 else 0.0
    compound = round(compound, 3)

    if compound >= 0.15:
        label = "Positive"
    elif compound <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    # Percentage breakdown
    if total_weight > 0:
        pos_pct = round((pos_score / (total_weight + 2.0)) * 100, 1)
        neg_pct = round((neg_score / (total_weight + 2.0)) * 100, 1)
        neu_pct = round(100.0 - pos_pct - neg_pct, 1)
    else:
        pos_pct = 15.0
        neg_pct = 10.0
        neu_pct = 75.0

    subjectivity = round(min(1.0, (subjective_words / max(1, len(tokens))) * 2.5), 2)

    # Dominant emotion
    max_emotion = max(emotions_count.items(), key=lambda x: x[1])
    dominant_emotion = max_emotion[0] if max_emotion[1] > 0 else ("Joy" if label == "Positive" else ("Sadness" if label == "Negative" else "Calm/Neutral"))

    # Extract non-stopword tokens for word frequency
    meaningful_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    return {
        "score": compound,
        "label": label,
        "compound": compound,
        "positive_pct": pos_pct,
        "neutral_pct": max(0.0, neu_pct),
        "negative_pct": neg_pct,
        "subjectivity": subjectivity,
        "emotions": emotions_count,
        "dominant_emotion": dominant_emotion,
        "keywords": list(set(meaningful_tokens[:8]))
    }


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names from various social media export formats."""
    df = df.copy()
    col_map = {}
    
    # Lowercase mapping lookup
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if c_clean in ["text", "caption", "content", "message", "post", "tweet", "description"]:
            col_map[col] = "content"
        elif c_clean in ["platform", "network", "channel", "source"]:
            col_map[col] = "platform"
        elif c_clean in ["likes", "like_count", "favorite_count", "favorites", "reactions"]:
            col_map[col] = "likes"
        elif c_clean in ["comments", "comment_count", "replies", "reply_count"]:
            col_map[col] = "comments"
        elif c_clean in ["shares", "share_count", "retweets", "retweet_count", "reposts", "repost_count"]:
            col_map[col] = "shares"
        elif c_clean in ["impressions", "views", "view_count", "reach"]:
            col_map[col] = "impressions"
        elif c_clean in ["clicks", "url_clicks", "link_clicks"]:
            col_map[col] = "clicks"
        elif c_clean in ["date", "timestamp", "created_at", "posted_at", "publish_date", "time"]:
            col_map[col] = "date"
        elif c_clean in ["type", "format", "post_type", "media_type"]:
            col_map[col] = "type"
        elif c_clean in ["author", "username", "handle", "creator", "page"]:
            col_map[col] = "author"

    df = df.rename(columns=col_map)

    # Ensure required columns exist with reasonable defaults
    if "content" not in df.columns:
        df["content"] = "Social media post update"
    if "platform" not in df.columns:
        df["platform"] = "Instagram"
    if "likes" not in df.columns:
        df["likes"] = 0
    if "comments" not in df.columns:
        df["comments"] = 0
    if "shares" not in df.columns:
        df["shares"] = 0
    if "impressions" not in df.columns:
        # Default impressions to estimated 10x total interactions
        df["impressions"] = (df["likes"] + df["comments"] + df["shares"]) * 10 + 100
    if "type" not in df.columns:
        df["type"] = "Image"
    if "author" not in df.columns:
        df["author"] = "Official Brand"
    if "date" not in df.columns:
        now = datetime.now()
        df["date"] = [(now - timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(len(df))]

    # Numeric conversion
    for col in ["likes", "comments", "shares", "impressions"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Calculate engagement metrics
    df["total_engagements"] = df["likes"] + df["comments"] + df["shares"]
    df["engagement_rate"] = np.where(
        df["impressions"] > 0,
        np.round((df["total_engagements"] / df["impressions"]) * 100, 2),
        0.0
    )

    # Standardize platform names
    df["platform"] = df["platform"].astype(str).str.strip().str.title()
    platform_alias = {
        "Twitter": "X / Twitter",
        "X": "X / Twitter",
        "Twitter/X": "X / Twitter",
        "X/Twitter": "X / Twitter",
        "Ig": "Instagram",
        "Yt": "YouTube",
        "Youtube": "YouTube",
        "Linkedin": "LinkedIn",
        "Tiktok": "TikTok",
        "Fb": "Facebook",
        "Facebook": "Facebook"
    }
    df["platform"] = df["platform"].replace(platform_alias)

    return df


def compute_overview_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive KPIs, platform breakdown, time series, and sentiment distributions."""
    if df.empty:
        return {}

    total_posts = len(df)
    total_impressions = int(df["impressions"].sum())
    total_likes = int(df["likes"].sum())
    total_comments = int(df["comments"].sum())
    total_shares = int(df["shares"].sum())
    total_engagements = int(df["total_engagements"].sum())
    
    avg_engagement_rate = round(float(df["engagement_rate"].mean()), 2)
    max_engagement_rate = round(float(df["engagement_rate"].max()), 2)

    # Run sentiment analysis on all post texts
    sentiments = [analyze_sentiment_text(str(text)) for text in df["content"]]
    df["sentiment_score"] = [s["compound"] for s in sentiments]
    df["sentiment_label"] = [s["label"] for s in sentiments]
    df["dominant_emotion"] = [s["dominant_emotion"] for s in sentiments]

    sentiment_counts = df["sentiment_label"].value_counts().to_dict()
    sentiment_distribution = {
        "Positive": int(sentiment_counts.get("Positive", 0)),
        "Neutral": int(sentiment_counts.get("Neutral", 0)),
        "Negative": int(sentiment_counts.get("Negative", 0))
    }

    # Emotion distribution
    emotion_counts = df["dominant_emotion"].value_counts().to_dict()

    # Platform breakdown
    platform_groups = df.groupby("platform").agg(
        posts=("content", "count"),
        impressions=("impressions", "sum"),
        engagements=("total_engagements", "sum"),
        likes=("likes", "sum"),
        comments=("comments", "sum"),
        shares=("shares", "sum"),
        avg_engagement_rate=("engagement_rate", "mean"),
        avg_sentiment=("sentiment_score", "mean")
    ).reset_index()

    platform_stats = []
    for _, row in platform_groups.iterrows():
        platform_stats.append({
            "platform": str(row["platform"]),
            "posts": int(row["posts"]),
            "impressions": int(row["impressions"]),
            "engagements": int(row["engagements"]),
            "likes": int(row["likes"]),
            "comments": int(row["comments"]),
            "shares": int(row["shares"]),
            "avg_engagement_rate": round(float(row["avg_engagement_rate"]), 2),
            "avg_sentiment": round(float(row["avg_sentiment"]), 2)
        })

    # Content type breakdown (Video, Reel, Carousel, Image, Text)
    type_groups = df.groupby("type").agg(
        posts=("content", "count"),
        engagements=("total_engagements", "sum"),
        avg_engagement_rate=("engagement_rate", "mean")
    ).reset_index()
    
    content_types = [
        {
            "type": str(r["type"]).title(),
            "posts": int(r["posts"]),
            "engagements": int(r["engagements"]),
            "avg_engagement_rate": round(float(r["avg_engagement_rate"]), 2)
        }
        for _, r in type_groups.iterrows()
    ]

    # Time series aggregation
    # Parse dates safely
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = df.dropna(subset=["parsed_date"]).sort_values("parsed_date")

    time_series = []
    if not valid_dates.empty:
        # Group by daily period
        valid_dates["day_str"] = valid_dates["parsed_date"].dt.strftime("%Y-%m-%d")
        daily = valid_dates.groupby("day_str").agg(
            impressions=("impressions", "sum"),
            engagements=("total_engagements", "sum"),
            posts=("content", "count"),
            avg_sentiment=("sentiment_score", "mean")
        ).reset_index()

        for _, r in daily.iterrows():
            time_series.append({
                "date": str(r["day_str"]),
                "impressions": int(r["impressions"]),
                "engagements": int(r["engagements"]),
                "posts": int(r["posts"]),
                "avg_sentiment": round(float(r["avg_sentiment"]), 2)
            })

    # Top performing posts
    top_posts = df.sort_values("total_engagements", ascending=False).head(10)
    top_posts_list = []
    for _, r in top_posts.iterrows():
        top_posts_list.append({
            "platform": str(r["platform"]),
            "author": str(r["author"]),
            "content": str(r["content"]),
            "date": str(r["date"]),
            "type": str(r["type"]),
            "likes": int(r["likes"]),
            "comments": int(r["comments"]),
            "shares": int(r["shares"]),
            "impressions": int(r["impressions"]),
            "engagement_rate": round(float(r["engagement_rate"]), 2),
            "sentiment": str(r["sentiment_label"])
        })

    # Extract Top Hashtags
    hashtags_dict = {}
    for text in df["content"]:
        tags = re.findall(r"#\w+", str(text).lower())
        for tag in tags:
            hashtags_dict[tag] = hashtags_dict.get(tag, 0) + 1
    
    sorted_hashtags = sorted(hashtags_dict.items(), key=lambda x: x[1], reverse=True)[:15]
    top_hashtags = [{"tag": tag, "count": count} for tag, count in sorted_hashtags]

    # Best Posting Hour / Day of Week
    best_posting_times = {}
    if not valid_dates.empty:
        valid_dates["hour"] = valid_dates["parsed_date"].dt.hour
        valid_dates["weekday"] = valid_dates["parsed_date"].dt.day_name()
        
        # Day of week average engagement
        dow_agg = valid_dates.groupby("weekday")["engagement_rate"].mean().to_dict()
        hour_agg = valid_dates.groupby("hour")["engagement_rate"].mean().to_dict()
        
        best_day = max(dow_agg.items(), key=lambda x: x[1])[0] if dow_agg else "Wednesday"
        best_hour = max(hour_agg.items(), key=lambda x: x[1])[0] if hour_agg else 18
    else:
        best_day = "Wednesday"
        best_hour = 18

    return {
        "kpis": {
            "total_posts": total_posts,
            "total_impressions": total_impressions,
            "total_engagements": total_engagements,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "avg_engagement_rate": avg_engagement_rate,
            "max_engagement_rate": max_engagement_rate,
            "best_posting_day": best_day,
            "best_posting_hour": f"{best_hour:02d}:00"
        },
        "platforms": platform_stats,
        "content_types": content_types,
        "sentiment_distribution": sentiment_distribution,
        "emotion_distribution": emotion_counts,
        "time_series": time_series,
        "top_posts": top_posts_list,
        "top_hashtags": top_hashtags
    }


def optimize_draft_caption(caption: str, platform: str = "Instagram", target_hour: int = 14) -> Dict[str, Any]:
    """
    Evaluates a draft social media post and provides:
    - Predicted engagement score (0 - 100)
    - Sentiment score and breakdown
    - Readability & length checks
    - Actionable recommendations (hashtags, length, tone, call to action, emoji presence)
    """
    char_len = len(caption)
    word_count = len(caption.split())
    hashtags = re.findall(r"#\w+", caption)
    mentions = re.findall(r"@\w+", caption)
    emoji_count = len(re.findall(r"[\U00010000-\U0010ffff]", caption))
    has_cta = any(phrase in caption.lower() for phrase in [
        "link in bio", "comment below", "let us know", "what do you think",
        "share this", "tag a friend", "tap to shop", "swipe up", "visit", "subscribe",
        "check out", "tell us", "click link", "save for later"
    ])
    has_question = "?" in caption

    # Sentiment
    sent_res = analyze_sentiment_text(caption)
    sentiment_score = sent_res["compound"]

    # Platform-specific optimal lengths
    platform_specs = {
        "Instagram": {"opt_min": 70, "opt_max": 250, "ideal_tags": (3, 8), "best_hours": [11, 14, 18, 20]},
        "X / Twitter": {"opt_min": 50, "opt_max": 240, "ideal_tags": (1, 3), "best_hours": [9, 12, 17]},
        "LinkedIn": {"opt_min": 100, "opt_max": 500, "ideal_tags": (2, 5), "best_hours": [8, 10, 13]},
        "YouTube": {"opt_min": 150, "opt_max": 600, "ideal_tags": (3, 7), "best_hours": [15, 17, 19]},
        "TikTok": {"opt_min": 40, "opt_max": 180, "ideal_tags": (3, 6), "best_hours": [14, 19, 21]}
    }

    spec = platform_specs.get(platform, platform_specs["Instagram"])
    
    # Calculate Engagement Score components (out of 100)
    score = 50.0  # baseline

    # Length evaluation
    if spec["opt_min"] <= char_len <= spec["opt_max"]:
        score += 15.0
        length_feedback = "Optimal length for this platform."
    elif char_len < spec["opt_min"]:
        score -= 10.0
        length_feedback = f"A bit too brief. Consider elaborating with more context (aim for {spec['opt_min']}+ chars)."
    else:
        score -= 5.0
        length_feedback = f"Slightly lengthy. Consider tightening the message for punchier impact (aim for <{spec['opt_max']} chars)."

    # Hashtag evaluation
    min_tags, max_tags = spec["ideal_tags"]
    tag_count = len(hashtags)
    if min_tags <= tag_count <= max_tags:
        score += 12.0
        tag_feedback = f"Great hashtag balance ({tag_count} tags)."
    elif tag_count == 0:
        score -= 8.0
        tag_feedback = f"No hashtags found. Add {min_tags}-{max_tags} relevant hashtags to boost discoverability."
    elif tag_count > max_tags:
        score -= 5.0
        tag_feedback = f"High hashtag count ({tag_count}). Avoid looking spammy by trimming to {max_tags} tags."
    else:
        score += 5.0
        tag_feedback = f"Consider adding {min_tags - tag_count} more specific tag(s) to maximize reach."

    # CTA & Interactivity
    if has_cta:
        score += 10.0
    if has_question:
        score += 8.0

    # Emoji presence
    if emoji_count > 0:
        score += 5.0
    
    # Sentiment bonus
    if sentiment_score > 0.2:
        score += 8.0
    elif sentiment_score < -0.3:
        score -= 5.0

    # Timing check
    timing_bonus = 0
    if target_hour in spec["best_hours"]:
        timing_bonus = 7.0
        score += timing_bonus
        timing_feedback = f"Prime posting hour for {platform} ({target_hour:02d}:00)."
    else:
        timing_feedback = f"Suggested peak hours for {platform} are: {', '.join([f'{h}:00' for h in spec['best_hours']])}."

    # Clamp score
    final_score = int(min(100.0, max(15.0, score)))

    # Actionable suggestions list
    recommendations = []
    if not has_cta:
        recommendations.append("Include a clear Call-to-Action (e.g., 'Comment your thoughts', 'Link in bio', 'Share this tip').")
    if not has_question:
        recommendations.append("Ask a captivating question to spark discussions in the comments.")
    if tag_count < min_tags:
        recommendations.append(f"Add relevant hashtags to reach wider communities (target: {min_tags}-{max_tags} tags).")
    if emoji_count == 0:
        recommendations.append("Incorporate 1-2 expressive emojis to make the visual hierarchy pop.")
    if sentiment_score <= 0.05:
        recommendations.append("Infuse more energetic, positive or conversational vocabulary to raise emotional connection.")

    # High-performing hashtag recommendations based on platform
    suggested_hashtags = {
        "Instagram": ["#SocialGrowth", "#ContentCreator", "#DigitalMarketing", "#TrendAlert", "#DailyInspiration"],
        "X / Twitter": ["#TechTrends", "#BuildInPublic", "#MarketingStrategy", "#TrendingNow"],
        "LinkedIn": ["#Leadership", "#Innovation", "#ProfessionalGrowth", "#IndustryInsights", "#Networking"],
        "YouTube": ["#CreatorEconomy", "#VideoTutorial", "#SubscribeNow", "#TechReview"],
        "TikTok": ["#ViralVideo", "#ForYouPage", "#Trends2024", "#CreatorTips"]
    }.get(platform, ["#SocialMedia", "#Viral", "#Engagement", "#Strategy"])

    return {
        "engagement_score": final_score,
        "score_grade": "A+" if final_score >= 85 else ("A" if final_score >= 75 else ("B" if final_score >= 60 else "C")),
        "char_count": char_len,
        "word_count": word_count,
        "hashtag_count": tag_count,
        "emoji_count": emoji_count,
        "has_cta": has_cta,
        "has_question": has_question,
        "length_feedback": length_feedback,
        "tag_feedback": tag_feedback,
        "timing_feedback": timing_feedback,
        "sentiment": sent_res,
        "recommendations": recommendations,
        "suggested_hashtags": suggested_hashtags
    }
