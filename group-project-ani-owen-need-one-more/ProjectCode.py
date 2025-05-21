import os
import pandas as pd
import random
import openai
from openai import OpenAI

"""
DO NOT RUN THIS CODE AGAIN. IT WILL REGENERATE
EVERYTHING AND OVERWRITE THE FILES.
"""
# Manual environment variable read
_open_ai_tkn = os.getenv("OPENAI_KEY")
_project_tkn = os.environ.get('OPENAI_PROJECT')
_organisation_tkn = os.environ.get('OPENAI_ORG')

# Client instantiation
client = OpenAI(
    organization=_organisation_tkn,
    project=_project_tkn,
    api_key=_open_ai_tkn
)


# Out-of-scope grammar concepts (strictly N3 or higher)
OUT_OF_SCOPE_GRAMMAR_POOL = [
    "〜ことにする", "〜ようにする", "〜てしまう", "〜ば〜ほど", "〜ないことはない", "〜ことがある", "〜たばかり", "〜ておく", "〜てみる", "〜ずに", 
    "〜ということだ", "〜ようだ", "〜らしい", "〜に違いない", "〜ながらも", "〜たところ", "〜ことなく", "〜わけではない", "〜ことにしている", "〜たとたん", 
    "〜にしては", "〜にしても", "〜わけにはいかない", "〜ことは〜が", "〜たまま", "〜つもり"
]

# Curated JLPT N4 vocab pool (150 words, balanced by type)
N4_VOCAB_POOL = [
    "犬", "猫", "本", "学校", "先生", "友達", "朝", "昼", "夜", "家",
    "道", "駅", "店", "電話", "車", "食べ物", "飲み物", "海", "山", "川",
    "勉強", "運動", "趣味", "音楽", "映画", "旅行", "問題", "答え", "時間", "写真",
    "読む", "書く", "話す", "聞く", "見る", "歩く", "走る", "座る", "立つ", "遊ぶ",
    "食べる", "飲む", "買う", "売る", "作る", "使う", "始める", "終わる", "分かる", "待つ",
    "楽しい", "悲しい", "うれしい", "怖い", "忙しい", "静か", "にぎやか", "寒い", "暑い", "高い",
    "安い", "大きい", "小さい", "新しい", "古い", "近い", "遠い", "良い", "悪い", "早い",
    "遅い", "明るい", "暗い", "簡単", "難しい", "便利", "不便", "上手", "下手", "有名",
    "一緒に", "よく", "たくさん", "少し", "とても", "まだ", "もう", "いつも", "時々", "最近",
    "昨日", "今日", "明日", "今朝", "今晩", "来週", "先週", "今月", "来月", "去年",
    "毎日", "毎週", "毎月", "毎年", "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日",
    "日曜日", "午前", "午後", "朝ごはん", "昼ごはん", "晩ごはん", "水", "お茶", "牛乳", "ビール",
    "新聞", "雑誌", "手紙", "鍵", "財布", "靴", "帽子", "洋服", "部屋", "建物"
]

# Out-of-scope vocab (N3 or higher examples)
OUT_OF_SCOPE_VOCAB_POOL = [
    "概念", "促進", "矛盾", "充実", "徹底", "抽象的", "模倣", "推論", "肯定", "改善",
    "承認", "傾向", "主観", "客観", "多様性", "分析", "統計", "成長", "影響", "経済",
    "政治", "技術", "環境", "制度", "社会", "文化", "責任", "論理", "理論", "仮定"
]

NAMES_POOL = [
        "たろう", "はなこ", "けんじ", "さくら", "ゆうた", "あや", "しょうた", "りな",
        "だいち", "みさき", "こうた", "ゆか", "りく", "えみ", "そうた", "なお",
        "かずき", "ひな", "しゅん", "まい", "ゆうき", "みお", "はるき", "のぞみ",
        "けい", "あい", "まこと", "ひろし", "ともこ", "かおる", "まさし", "えりか",
        "たかし", "あすか", "ゆり", "ひろこ", "まさと", "みゆき", "けん", "みさと",
        "しんじ", "さとし", "なおき", "ちひろ", "あきら", "さやか", "ともや", "まなみ"
    ]

THEMES_POOL = [
    '友情', '学校生活', '家族', '冒険', 'スポーツ',
    '自然', '買い物', '食べ物', '旅行', '夢'
]

if not os.path.exists('dataset.csv'):
    data = {
        'ID': list(range(1, 101)),
        'Characters': [", ".join(random.sample(NAMES_POOL, 2)) for _ in range(100)],
        'Theme': [random.choice(THEMES_POOL) for _ in range(100)],
        'Vocab': ["、".join(random.sample(N4_VOCAB_POOL, 3) + random.sample(OUT_OF_SCOPE_VOCAB_POOL, 1)) for _ in range(100)],
        'Target_Length': [random.choice([200, 225, 250, 275, 300]) for _ in range(100)],
        'OutOfScopeGrammar': [random.choice(OUT_OF_SCOPE_GRAMMAR_POOL) for _ in range(100)]
    }
    pd.DataFrame(data).to_csv("dataset.csv", index=False)
    print("Dataset generated and saved to dataset.csv")

# Prompt abstraction logic
def generate_prompt(row, abstraction_id):
    if abstraction_id == 'A1':
        return f"""次のキャラクターとテーマを使って、短い日本語の物語を書いてください。

キャラクター: {row['Characters']}
テーマ: 「{row['Theme']}」

・JLPT N4レベル以下の語彙、文法、漢字のみを使用してください。
・300文字以内にしてください。
・説明文や前置きは不要です。物語本文のみを書いてください。"""

    elif abstraction_id == 'A2':
        return f"""Please write a short story in Japanese using the following theme and characters.

Theme: "{row['Theme']}"
Characters: {row['Characters']}

- The story must be in Japanese only.
- Use only vocabulary, grammar, and kanji from JLPT N4 or below.
- Keep it under 300 characters.
- Do not include any introductory or explanatory text — only the story body."""

    elif abstraction_id == 'A3':
        return f"""以下の語彙（N4 + 高度な語彙1語）をすべて使って、日本語の物語を書いてください。

語彙: {row['Vocab']}

・JLPT N4レベル以下の語彙、文法、漢字のみを使用してください。
・次の高度な文法も1回だけ使用しても構いません: {row['OutOfScopeGrammar']}
・300文字以内にしてください。
・説明や前置きは書かず、物語本文のみを出力してください。"""

    elif abstraction_id == 'A4':
        return f"""日本語で短い物語を書いてください。

・目標文字数はおよそ{row['Target_Length']}文字です。できるだけ正確に合わせてください。
・JLPT N4レベル以下の語彙、文法、漢字のみを使用してください。
・他のテキスト（説明や挨拶など）は不要です。物語本文のみを書いてください。"""

    else:
        raise ValueError("Invalid abstraction ID")

# Story generation
results = []
df = pd.read_csv("dataset.csv")

total = len(df) * 4
counter = 1

for _, row in df.iterrows():
    for abstraction_id in ['A1', 'A2', 'A3', 'A4']:
        prompt = generate_prompt(row, abstraction_id)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        story = response.choices[0].message.content.strip()
        results.append({
            "ID": row["ID"],
            "Abstraction": abstraction_id,
            "Prompt": prompt,
            "Story": story
        })
        print(f"Completed {counter}/{total}")
        counter += 1

pd.DataFrame(results).to_csv("all_generated_outputs.csv", index=False)
print("Story generation complete.")

df_all = pd.read_csv("all_generated_outputs.csv")
final_eval_samples = (
    df_all.groupby("Abstraction")
    .apply(lambda x: x.sample(n=15, random_state=42))
    .reset_index(drop=True)
)
final_eval_samples.to_csv("final_evaluation_sample.csv", index=False)
print("Final evaluation sample of 60 stories saved to final_evaluation_sample.csv")

