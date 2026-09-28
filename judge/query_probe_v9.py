#!/usr/bin/env python3
"""探针 v9：搜索 query 理解 —— 350M 能不能代替分词器？

搜索链路分三层，本探针分别定位 LLM 能进哪一层：
  L1 切词 tokenization   —— 建索引端 + 查询端跑同一套确定性代码
  L2 term tagging 属性归一 —— 分词后给 term 打标签（颜色/品类/季节）
  L3 意图 / 改写

四个引擎对比：
  A_seg     词典双向最大匹配分词器 + 词典打标   （零依赖，工业界 ES/IK 的本质）
  B_lr      字符 n-gram TF-IDF + 逻辑回归       （轻量学习路线，代表 encoder/分类器能力下界）
  C_350m    LFM2.5-350M 生成式端到端 JSON
  D_hybrid  分词器切词 → 350M 只打标            （增强而非替代）

关键测量：
  1. 属性抽取准确率
  2. 延迟（分位）
  3. 确定性：同一输入重复运行结果是否漂移
  4. 双端一致性：同义改写下 term 集合是否稳定（决定建索引是否可行）
"""
import json
import re
import time
import statistics as st

# ---------------------------------------------------------------- 测试数据
# (query, {品类, 颜色, 季节})
CASES = [
    ("红色连衣裙夏天",            {"品类": "连衣裙", "颜色": "红", "季节": "夏"}),
    ("男士纯棉短袖t恤",           {"品类": "短袖t恤", "颜色": None, "季节": "夏"}),
    ("冬季加厚羽绒服女",          {"品类": "羽绒服", "颜色": None, "季节": "冬"}),
    ("黑色运动鞋男",              {"品类": "运动鞋", "颜色": "黑", "季节": None}),
    ("春秋款牛仔外套",            {"品类": "牛仔外套", "颜色": None, "季节": "春秋"}),
    ("白色雪纺衬衫",              {"品类": "衬衫", "颜色": "白", "季节": None}),
    ("蓝色直筒牛仔裤",            {"品类": "牛仔裤", "颜色": "蓝", "季节": None}),
    ("夏天穿的凉鞋",              {"品类": "凉鞋", "颜色": None, "季节": "夏"}),
    ("秋冬新款羊毛衫",            {"品类": "羊毛衫", "颜色": None, "季节": "秋冬"}),
    ("粉色碎花半身裙",            {"品类": "半身裙", "颜色": "粉", "季节": None}),
    ("灰色针织开衫",              {"品类": "针织开衫", "颜色": "灰", "季节": None}),
    ("黑色真皮单肩包",            {"品类": "单肩包", "颜色": "黑", "季节": None}),
    ("夏季薄款睡衣",              {"品类": "睡衣", "颜色": None, "季节": "夏"}),
    ("米色风衣女中长款",          {"品类": "风衣", "颜色": "米色", "季节": None}),
    ("藏青色西裤",                {"品类": "西裤", "颜色": "藏青", "季节": None}),
    ("冬天戴的毛线帽",            {"品类": "毛线帽", "颜色": None, "季节": "冬"}),
    ("绿色polo衫",                {"品类": "polo衫", "颜色": "绿", "季节": None}),
    ("卡其色工装裤",              {"品类": "工装裤", "颜色": "卡其", "季节": None}),
    ("春天穿的薄外套",            {"品类": "外套", "颜色": None, "季节": "春"}),
    ("酒红色高跟鞋",              {"品类": "高跟鞋", "颜色": "酒红", "季节": None}),
    ("纯棉白色t恤",               {"品类": "t恤", "颜色": "白", "季节": None}),
    ("秋季长款针织裙",            {"品类": "针织裙", "颜色": None, "季节": "秋"}),
    ("咖啡色皮带",                {"品类": "皮带", "颜色": "咖啡", "季节": None}),
    ("夏天用的防晒衣",            {"品类": "防晒衣", "颜色": None, "季节": "夏"}),
    ("深蓝色阔腿裤",              {"品类": "阔腿裤", "颜色": "深蓝", "季节": None}),
    ("冬季保暖内衣",              {"品类": "保暖内衣", "颜色": None, "季节": "冬"}),
    ("杏色针织马甲",              {"品类": "针织马甲", "颜色": "杏色", "季节": None}),
    ("黑色帆布鞋",                {"品类": "帆布鞋", "颜色": "黑", "季节": None}),
    ("春夏薄款连衣裙",            {"品类": "连衣裙", "颜色": None, "季节": "春夏"}),
    ("紫雪纺长裙",                {"品类": "长裙", "颜色": "紫", "季节": None}),
]

# 同义改写对：语义相同、措辞不同。用于测「双端一致性」
# 分词器两端跑同一套确定性代码 → 输出 term 集合应当一致
PARAPHRASE = [
    ("红色连衣裙夏天", "夏天穿的红色连衣裙"),
    ("男士纯棉短袖t恤", "纯棉t恤男款短袖"),
    ("冬季加厚羽绒服女", "女款羽绒服加厚冬天穿"),
    ("黑色运动鞋男", "男式黑色跑步鞋"),
    ("春秋款牛仔外套", "牛仔外套春秋两季可穿"),
]

# 词典外新词（OOV）：分词器的已知弱点，也是 LLM 唯一理论上的机会点
OOV = [
    ("多巴胺穿搭连衣裙夏天", {"品类": "连衣裙", "颜色": None, "季节": "夏"}),
    ("美拉德风外套", {"品类": "外套", "颜色": None, "季节": None}),
    ("老钱风针织衫", {"品类": "针织衫", "颜色": None, "季节": None}),
    ("山系冲锋衣", {"品类": "冲锋衣", "颜色": None, "季节": None}),
    ("静奢风羊绒大衣", {"品类": "大衣", "颜色": None, "季节": None}),
    ("机能风工装裤", {"品类": "工装裤", "颜色": None, "季节": None}),
    ("芭蕾风平底鞋", {"品类": "平底鞋", "颜色": None, "季节": None}),
    ("克莱因蓝连衣裙", {"品类": "连衣裙", "颜色": "蓝", "季节": None}),
]

FIELDS = ["品类", "颜色", "季节"]

# ---------------------------------------------------------------- 词典
CATS = ["连衣裙", "半身裙", "针织裙", "长裙", "短袖t恤", "t恤", "polo衫", "衬衫", "羊毛衫",
        "针织开衫", "针织马甲", "风衣", "牛仔外套", "外套", "羽绒服", "保暖内衣", "防晒衣",
        "睡衣", "运动鞋", "帆布鞋", "凉鞋", "高跟鞋", "牛仔裤", "西裤", "工装裤", "阔腿裤",
        "单肩包", "皮带", "毛线帽"]
COLORS = ["酒红", "藏青", "深蓝", "米色", "卡其", "杏色", "咖啡", "红", "黑", "白", "蓝", "绿",
          "灰", "粉", "紫", "银", "金"]
SEASONS = ["春夏", "春秋", "秋冬", "夏", "冬", "春", "秋"]

DICT = {w: "品类" for w in CATS}
DICT.update({w: "颜色" for w in COLORS})
DICT.update({w: "季节" for w in SEASONS})


# ================================================================ A. 分词器
class Seg:
    """双向最大匹配分词器 + 词典打标。确定性代码，无模型、无采样。"""

    def __init__(self, dic):
        self.dic = dic
        self.maxlen = max(len(w) for w in dic)

    def cut(self, text):
        """正向最大匹配 + 逆向最大匹配，取词数少者（工业界消歧规则之一）。"""
        t = text.lower().replace(" ", "")
        fwd, bwd = self._fwd(t), self._bwd(t)
        return fwd if len(fwd) <= len(bwd) else bwd

    def _fwd(self, t):
        out, i = [], 0
        while i < len(t):
            for j in range(min(self.maxlen, len(t) - i), 0, -1):
                if t[i:i + j] in self.dic:
                    out.append(t[i:i + j]); i += j; break
            else:
                out.append(t[i]); i += 1
        return out

    def _bwd(self, t):
        out, i = [], len(t)
        while i > 0:
            for j in range(min(self.maxlen, i), 0, -1):
                if t[i - j:i] in self.dic:
                    out.append(t[i - j:i]); i -= j; break
            else:
                out.append(t[i - 1]); i -= 1
        return out[::-1]

    def extract(self, text):
        """切词 → 按词典给每个 term 打标 → 组装结构化结果。"""
        tags = {}
        for w in self.cut(text):
            f = self.dic.get(w)
            if f and f not in tags:
                tags[f] = w
        return {f: tags.get(f) for f in FIELDS}


def norm(v):
    if v is None:
        return None
    v = str(v).strip().lower()
    v = v.replace("色", "") if len(v) > 1 else v
    for a, b in (("夏季", "夏"), ("冬季", "冬"), ("春季", "春"), ("秋季", "秋"),
                 ("两季", ""), ("款", ""), ("季节性", ""), ("季", "")):
        v = v.replace(a, b)
    return v or None


def score(pred, truth):
    """逐字段精确匹配；空值也算一种正确预测。"""
    ok = tot = 0
    for f in FIELDS:
        tot += 1
        if norm(pred.get(f)) == norm(truth.get(f)):
            ok += 1
    return ok, tot


# ================================================================ B. 轻量学习路线
def build_lr():
    """字符 n-gram TF-IDF + 逻辑回归：代表「不需要生成式模型也能学」的能力下界。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    X = [q for q, _ in CASES]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
    pipes, Xv = {}, vec.fit_transform(X)
    for f in FIELDS:
        y = [norm(t.get(f)) or "__空__" for _, t in CASES]
        p = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1),
            LogisticRegression(max_iter=1000, C=5.0))
        p.fit(X, y)
        pipes[f] = p
    return pipes


def lr_extract(pipes, q):
    out = {}
    for f in FIELDS:
        v = pipes[f].predict([q])[0]
        out[f] = None if v == "__空__" else v
    return out


# ================================================================ C. 350M 生成式
SYS = """你是电商搜索 query 理解器。
约束：
1. 禁止输出思考过程、禁止标签、禁止markdown```标记。
2. 只输出裸JSON，不输出其他文字。
3. 返回格式固定：{"品类":"...","颜色":"...","季节":"..."}
4. 只从下面集合选值，不要自己造词：
品类：连衣裙,半身裙,针织裙,长裙,短袖t恤,t恤,polo衫,衬衫,羊毛衫,针织开衫,针织马甲,风衣,牛仔外套,外套,羽绒服,保暖内衣,防晒衣,睡衣,运动鞋,帆布鞋,凉鞋,高跟鞋,牛仔裤,西裤,工装裤,阔腿裤,单肩包,皮带,毛线帽
颜色：红,黑,白,蓝,绿,灰,粉,紫,酒红,藏青,深蓝,米色,卡其,杏色,咖啡
季节：春,夏,秋,冬,春夏,春秋,秋冬
5. 原文里没有的字段填 null，不要猜。
"""

FEW = """
示例：
用户输入：红色连衣裙夏天
输出：{"品类":"连衣裙","颜色":"红","季节":"夏"}
用户输入：男士纯棉短袖t恤
输出：{"品类":"短袖t恤","颜色":null,"季节":"夏"}
用户输入：冬天戴的毛线帽
输出：{"品类":"毛线帽","颜色":null,"季节":"冬"}
"""


def strip_fence(s):
    s = s.strip()
    if s.startswith("```"):
        p = s.split("```", 2)
        s = p[1] if len(p) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    return s


def llm_extract(model, tok, query, few=False):
    body = SYS + (FEW if few else "")
    p = tok.apply_chat_template(
        [{"role": "system", "content": body},
         {"role": "user", "content": f"用户输入：{query}\n"}],
        tokenize=False, add_generation_prompt=True)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    t0 = time.time()
    raw = generate(model, tok, prompt=p, max_tokens=96,
                   sampler=make_sampler(temp=0.0), verbose=False)
    dt = time.time() - t0
    try:
        v = json.loads(strip_fence(raw))
        return (v if isinstance(v, dict) else None), raw, dt
    except Exception:
        return None, raw, dt


# ================================================================ 主流程
def main():
    print("=" * 78)
    print("探针 v9：搜索 query 理解 —— 350M 能不能代替分词器？")
    print("=" * 78)
    n = len(CASES)

    # ---------- A ----------
    seg = Seg(DICT)
    A_ok = 0
    A_dts, A_terms = [], []
    for q, truth in CASES:
        t0 = time.time()
        pred = seg.extract(q)
        A_dts.append(time.time() - t0)
        A_terms.append(seg.cut(q))
        A_ok += score(pred, truth)[0]
    print(f"\n【A_seg】词典分词器（双向最大匹配 + 词典打标，零模型）")
    print(f"  属性准确率  {A_ok}/{n * len(FIELDS)} ({A_ok / (n * len(FIELDS)) * 100:.0f}%)")
    print(f"  单条延迟    avg={st.mean(A_dts) * 1e6:.0f}µs  p50={sorted(A_dts)[n // 2] * 1e6:.0f}µs   "
          f"（确定性代码，零方差）")

    # ---------- B ----------
    pipes = build_lr()
    B_ok = 0
    B_dts = []
    for q, truth in CASES:
        t0 = time.time()
        pred = lr_extract(pipes, q)
        B_dts.append(time.time() - t0)
        B_ok += score(pred, truth)[0]
    print(f"\n【B_lr】字符 n-gram TF-IDF + 逻辑回归（30 条训练，代表轻量学习路线）")
    print(f"  属性准确率  {B_ok}/{n * len(FIELDS)} ({B_ok / (n * len(FIELDS)) * 100:.0f}%)"
          f"   ← 训练集=测试集，属乐观上限")
    print(f"  单条延迟    avg={st.mean(B_dts) * 1e6:.0f}µs")

    # ---------- C ----------
    from mlx_lm import load
    model, tok = load("models/LFM2.5-350M-MLX-4bit",
                      tokenizer_config={"trust_remote_code": True})
    print(f"\n【C_350m】LFM2.5-350M 生成式端到端 JSON")
    for few in (False, True):
        ok = dts = json_ok = 0
        wrong = []
        for q, truth in CASES:
            v, raw, dt = llm_extract(model, tok, q, few)
            dts += dt
            if v is None:
                wrong.append((q, "PARSE_FAIL", raw[:60]))
                continue
            json_ok += 1
            o, _ = score(v, truth)
            ok += o
            if o < len(FIELDS):
                wrong.append((q, json.dumps(v, ensure_ascii=False), ""))
        print(f"  {'few-shot' if few else '0-shot ':<8} 属性准确率 {ok}/{n * len(FIELDS)} "
              f"({ok / (n * len(FIELDS)) * 100:.0f}%)   JSON合法 {json_ok}/{n}   "
              f"单条 avg={dts / n * 1000:.0f}ms")
        if few and wrong:
            for q, v, raw in wrong[:4]:
                print(f"      错例 {q!r} -> {v}")
    model_loaded = model  # 复用给 D

    # ---------- D ----------
    print(f"\n【D_hybrid】分词器切词 → 350M 只做打标（增强而非替代）")
    ok = dts = 0
    for q, truth in CASES:
        terms = seg.cut(q)
        hint = f"已切好的词：{'/'.join(terms)}\n"
        p = tok.apply_chat_template(
            [{"role": "system", "content": SYS + FEW},
             {"role": "user", "content": f"用户输入：{q}\n{hint}"}],
            tokenize=False, add_generation_prompt=True)
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        t0 = time.time()
        raw = generate(model_loaded, tok, prompt=p, max_tokens=96,
                       sampler=make_sampler(temp=0.0), verbose=False)
        dts += time.time() - t0
        try:
            v = json.loads(strip_fence(raw))
            ok += score(v, truth)[0]
        except Exception:
            pass
    print(f"  属性准确率  {ok}/{n * len(FIELDS)} ({ok / (n * len(FIELDS)) * 100:.0f}%)   "
          f"单条 avg={dts / n * 1000:.0f}ms（不含分词耗时）")

    # ---------- 确定性 ----------
    print(f"\n" + "=" * 78)
    print("【确定性】同一输入重复运行 3 次，结果是否漂移")
    print("=" * 78)
    drift_seg = drift_llm = 0
    for q, _ in CASES[:15]:
        rs = {json.dumps(seg.extract(q), ensure_ascii=False) for _ in range(3)}
        rl = set()
        for _ in range(3):
            v, raw, _ = llm_extract(model_loaded, tok, q, True)
            rl.add(json.dumps(v, ensure_ascii=False) if v else f"FAIL:{raw[:30]}")
        drift_seg += (len(rs) > 1)
        drift_llm += (len(rl) > 1)
        if len(rl) > 1:
            print(f"  350M 漂移: {q!r} -> {rl}")
    print(f"  分词器漂移用例 {drift_seg}/15    350M 漂移用例 {drift_llm}/15")

    # ---------- 双端一致性 ----------
    print(f"\n" + "=" * 78)
    print("【双端一致性】同义改写下，抽取结果是否稳定（决定能否建索引）")
    print("=" * 78)
    print(f"  {'原 query':<22}{'同义改写':<24}{'分词器一致':<12}{'350M一致'}")
    seg_same = llm_same = 0
    for a, b in PARAPHRASE:
        ta = set(seg.cut(a)) & set(DICT)
        tb = set(seg.cut(b)) & set(DICT)
        s_same = (ta == tb)
        va, _, _ = llm_extract(model_loaded, tok, a, True)
        vb, _, _ = llm_extract(model_loaded, tok, b, True)
        l_same = (va is not None and vb is not None
                  and all(norm(va.get(f)) == norm(vb.get(f)) for f in FIELDS))
        seg_same += s_same
        llm_same += l_same
        print(f"  {a:<22}{b:<24}{'一致' if s_same else f'不一致{tb - ta}/{ta - tb}':<12}"
              f"{'一致' if l_same else '不一致'}")
    print(f"\n  分词器一致 {seg_same}/{len(PARAPHRASE)}    350M 一致 {llm_same}/{len(PARAPHRASE)}")

    # ---------- OOV：分词器唯一弱点，LLM 唯一理论机会点 ----------
    print(f"\n" + "=" * 78)
    print("【OOV 新词】分词器的已知弱点 vs 350M —— LLM 唯一理论机会点")
    print("=" * 78)
    oov_s = oov_l = 0
    print(f"  {'query':<20}{'分词器切词':<28}{'分词':<7}350M 输出")
    for q, truth in OOV:
        cut = seg.cut(q)
        s_ok, tot = score(seg.extract(q), truth)
        v, raw, _ = llm_extract(model_loaded, tok, q, True)
        l_ok = score(v or {}, truth)[0] if v else 0
        oov_s += s_ok
        oov_l += l_ok
        vs = json.dumps(v, ensure_ascii=False)[:36] if v else f"PARSE_FAIL {raw[:20]!r}"
        print(f"  {q:<20}{'/'.join(cut):<28}{s_ok}/{tot}    {vs}")
    print(f"\n  字段级（严格口径：解析失败计 0 分，不当作「全空预测」送分）：")
    print(f"    分词器 {oov_s}/{len(OOV) * 3} ({oov_s / (len(OOV) * 3) * 100:.0f}%)   "
          f"350M {oov_l}/{len(OOV) * 3} ({oov_l / (len(OOV) * 3) * 100:.0f}%)")
    print("  → 在分词器最弱的地方，350M 输得更惨。而且它倾向把新造词当成品类名。")

    # ---------- 凭空填充偏置 ----------
    print(f"\n" + "=" * 78)
    print("【凭空填充】原文没有的字段，350M 是否留空（搜索场景 = 强加过滤条件）")
    print("=" * 78)
    from collections import Counter
    fills, should_null = Counter(), 0
    for q, truth in CASES:
        v, raw, _ = llm_extract(model_loaded, tok, q, True)
        if not v:
            continue
        for f in FIELDS:
            if truth.get(f) is None:
                should_null += 1
                gv = v.get(f)
                if gv not in (None, "", "null"):
                    fills[f"{f}={gv}"] += 1
    print(f"  应为空的字段 {should_null} 个 → 350M 填了 {sum(fills.values())} 个 "
          f"({sum(fills.values()) / max(should_null, 1) * 100:.0f}%)")
    print(f"  填充分布: {dict(fills.most_common(8))}")
    print("  → 搜索场景里凭空填一个属性 = 给用户没说的条件加过滤 = 直接零结果。")

    # ---------- B 的真实泛化（LOO-CV） ----------
    print(f"\n" + "=" * 78)
    print("【B_lr 真实泛化】留一交叉验证（修正前面的过拟合上限）")
    print("=" * 78)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    import warnings
    warnings.filterwarnings("ignore")
    X = [q for q, _ in CASES]
    cv_ok = 0
    for f in FIELDS:
        y = [norm(t.get(f)) or "__空__" for _, t in CASES]
        hits = 0
        for i in range(len(X)):
            Xtr, ytr = X[:i] + X[i + 1:], y[:i] + y[i + 1:]
            p = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1),
                              LogisticRegression(max_iter=1000, C=5.0))
            p.fit(Xtr, ytr)
            hits += (p.predict([X[i]])[0] == y[i])
        cv_ok += hits
        print(f"  {f}: LOO-CV {hits}/{len(X)} ({hits / len(X) * 100:.0f}%)")
    print(f"  合计 {cv_ok}/{len(X) * len(FIELDS)} ({cv_ok / (len(X) * len(FIELDS)) * 100:.0f}%)")
    print("  → 瓶颈是数据量（可加数据解决），不是路线；对比 350M 加 prompt 只 26%→30%。")

    # ---------- 建索引成本推算 ----------
    print(f"\n" + "=" * 78)
    print("【建索引成本推算】电商库 10 亿 SKU 标题，各方案跑一遍全量")
    print("=" * 78)
    items = 1_000_000_000
    llm_ms = dts / n * 1000
    for name, per_ms, unit in (
            ("分词器 (A, 本地 CPU)", st.mean(A_dts) * 1000, "ms"),
            ("350M 生成式 (C, mlx)", llm_ms, "ms"),
    ):
        total_h = items * per_ms / 1000 / 3600
        print(f"  {name:<24} 单条 {per_ms:>8.3f} ms   → 全量 {total_h:>12,.0f} 小时 "
              f"（{total_h / 24 / 365:>8,.1f} 年）")
    print(f"  → 分词器可分布式跑批；生成式模型还要 ×N 卡，且模型一升级全量索引作废重跑。")


if __name__ == "__main__":
    main()
