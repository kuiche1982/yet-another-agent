#!/usr/bin/env python3
"""
探针 v8：350M 是否适合做「从长对话中抽取用户信息」

验证认知：抽取型任务 + 输出可程序校验 = 小模型主场象限。
用真实客服长对话（机票预订 / 退票全流程）做抽取目标，三变量对照：

  输入形态  A 全量原文(~6k字)  B 规则切片(~400字)  C 自然语言改写(~6k字)
  抽取引擎  regex 纯代码基线   /  350M 0-shot  /  350M few-shot
  输出校验  身份证长度+mod11-2 / 手机号正则 / 邮箱正则 / 日期解析 / 枚举白名单

运行： ./.venv/bin/python judge/pii_probe_v8.py
"""
import json
import re
import time
from datetime import datetime

# ────────────────────────────── 语料 ──────────────────────────────

TRANSCRIPT = """## 查询 8 月 10 日北京到上海航班

### 用户

[$travelkit-pro](~/.codex/skills/travelkit-pro/SKILL.md) 查一下8月10 北京到上海的航班 一个成人 经济舱

### 助手

查好了：8 月 10 日北京到上海，1 成人，经济舱。下面 5 个都是已验价、余位够的直飞方案，价格为含税总价。

| 方案 | 航程 | 航班号 | 日期 | 航段 | 时间 | 飞行时长 | 舱位 | 行李 | 价格 | 总价 | 供应渠道 |
| - | - | - | - | - | - | - | - | - | - | - | - |
| 1 | 直飞 | MU5231 | 8月10日 | PKX→PVG(T1) | 23:20-01:15(+1) | 1h55m | 经济 Z舱，余位9 | 手提1×8kg，托运1×20kg | ¥500/人 | ¥500/1人 | 已验价 |
| 2 | 直飞 | MU5186 | 8月10日 | PKX→PVG(T1) | 07:45-09:55 | 2h10m | 经济 Z舱，余位9 | 手提1×8kg，托运1×20kg | ¥600/人 | ¥600/1人 | 已验价 |
| 3 | 直飞 | HO1258 | 8月10日 | PKX→PVG(T2) | 11:40-13:55 | 2h15m | 经济 E舱，余位9 | 手提1×5kg，托运1×20kg | ¥700/人 | ¥700/1人 | 已验价 |
| 4 | 直飞 | MU9192 | 8月10日 | PKX→SHA(T2) | 20:45-23:10 | 2h25m | 经济 Z舱，余位9 | 手提1×8kg，托运1×20kg | ¥600/人 | ¥600/1人 | 已验价 |
| 5 | 直飞 | MU5126 | 8月10日 | PEK(T2)→SHA(T2) | 20:00-21:50 | 1h50m | 经济 T舱，余位9 | 手提1×8kg，托运1×20kg | ¥750/人 | ¥750/1人 | 已验价 |

我会优先推荐 方案 2：比最低价只贵 ¥100，但当天上午到，不用凌晨落地，行李也更好。
如果你更看重到虹桥，选 方案 4；如果你从首都机场出发更方便，选 方案 5。

看中哪个方案，直接说"订方案几"就行；也可以让我按时间段、只看虹桥、只看首都机场/大兴机场、或压预算重新筛。

## 查询方案一退改政策

### 用户

方案一的退改政策怎么样

### 助手

方案一 MU5231（8月10日 23:20 大兴→浦东，经济 Z舱，¥500）的退改政策如下：

| 距起飞时间 | 退票费 | 改签费 |
| - | -: | -: |
| 起飞前 30 天以上 | ¥0 | ¥0 |
| 起飞前 7–30 天 | ¥70 | ¥18 |
| 起飞前 2–7 天 | ¥140 | ¥105 |
| 起飞前 4小时–2天 | ¥245 | ¥175 |
| 起飞前 4小时内或起飞后 | ¥263 | ¥210 |

现在距离 8月10日 23:20 起飞还有 7 天以上、不满 30 天，所以如果现在退票，按当前规则看退票费是 ¥70；改签费是 ¥18。另：这个票没有"废票"处理。

## 订方案一

### 用户

那订方案一吧

### 助手

可以订。还需要你给我这几项资料，凑齐后我会把行程、乘客、联系人、价格、行李和退改规则完整列出来让你确认；你确认后我才会创建订单。

请发：

1. 乘机人姓名：姓、名
2. 性别：男/女
3. 出生日期：YYYY-MM-DD
4. 国籍：比如中国就写 CN
5. 证件类型和号码：中国居民身份证或护照；如果用护照，还要护照有效期
6. 联系人姓名
7. 联系人邮箱
8. 联系人手机号和区号：比如 +86 138xxxx0000

方案一基线：8月10日 MU5231，23:20 大兴 PKX 出发，8月11日 01:15 到浦东 PVG T1，经济 Z舱，含税 ¥500。

### 用户

@@PASSENGER_BLOCK@@

### 助手

资料齐了，请你核对一下。确认无误后，我会立刻重新验价；如果航班、价格、行李和退改规则都没变，就创建订单。创建订单后还不会自动付款。

| # | 类型 | 证件姓名 | 性别 / 出生日期 / 国籍 | 证件 | 校验 |
| --- | --- | --- | --- | --- | --- |
| 乘客1 | 成人 | 姓 测；名 试 | 男 / 1998-01-01 / CN | 身份证 110235199801013127 | 通过 |

联系人：测试，+86 13800000000，test@example.com

行程：MU5231，8月10日 23:20 大兴 PKX 出发，8月11日 01:15 到浦东 PVG T1，直飞，经济 Z舱，余位9。
价格：¥500/1人，含税。
行李：手提 1×8kg，托运 1×20kg。
退改：当前时间看，起飞前 7–30 天退票费 ¥70，改签费 ¥18；起飞越近费用会变化。

请回复"确认创建订单"。

### 用户

没问题

### 助手

订单已创建，尚未付款。

| 订单 | 订单号 | 状态 | 行程 | 人数 | 总价 | 付款截止 |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| 订单1 | FOL728823L | 待付款 | 2026-08-10 PKX → PVG(T1)，MU5231 | 1人 | CNY 500 | 2026-07-28 22:27:36 CST |

请注意：未在 2026-07-28 22:27:36 CST 前付款，订单会自动取消。

目前仅支持余额付款。要继续付款的话，请回复"确认余额付款"。

## 支付

### 用户

支付吧

### 助手

付款成功，订单 FOL728823L 已扣余额 CNY 500。

当前订单状态是 出票处理中，还不是已出票；我已经按规则查了一次，票号暂时还没生成。
行程：8月10日 MU5231，23:20 大兴 PKX 出发，8月11日 01:15 到浦东 PVG T1。

最新余额：CNY -1,310，额度 CNY 2,000。稍后你可以让我查出票进度或票号。

## 出票查询

### 用户

出票了嘛？

### 助手

还没完全出票。

订单 FOL728823L 当前仍是 出票处理中，票号暂时还没有生成。
航段 MU5231：8月10日 PKX → PVG(T1)，票联状态已显示可用迹象，但订单和票号状态还没最终完成，所以我不能把它算作"已出票"。

稍后再查一次比较稳。

### 用户

再查一下

### 助手

出票了。

订单 FOL728823L 已出票，票号：TEST01。
乘客：测/试
航班：MU5231，8月10日 23:20 PKX → 8月11日 01:15 PVG(T1)
订座记录：TX1827

票联目前是可使用状态。

## 退北京到上海订单

### 用户

帮我退票吧

### 助手

可以退。最新查询显示这张票已出票，乘客 测/试 目前可提交退票申请。

退票范围如下：

订单：FOL728823L
乘客：测/试
航段：MU5231，8月10日 23:20 PKX → 8月11日 01:15 PVG(T1)
票号：TEST01
退票原因：自愿退票

请注意：确认后我会先提交退票申请，申请需要供应商审核，不会立即退票。审核完成后，我会告诉你最终可以退回多少钱，再由你决定是否继续执行退票。

如果确认提交，请回复：确认提交退票申请。

### 用户

对，确认

### 助手

退票申请已提交。

申请编号：FRL728N33F
原订单号：FOL728823L
乘客：测/试
航段：MU5231，8月10日 23:20 PKX → 8月11日 01:15 PVG(T1)
原因：自愿退票

退票申请正在审核中，请耐心等待。审核完成后，你可以回来问我进度，我会再查最终可以退回多少钱，然后由你决定是否继续执行退票。

### 用户

我刚刚收到邮件了

### 助手

审核结果出来了。

审核后的实际可退总额：CNY 500

原订单：FOL728823L
退票申请编号：FRL728N33F
退票原因：自愿退票

| 退票组 | 乘客 | 退票航段 | 退票费 | 服务费 | 实际可退 |
| --- | --- | --- | --- | ---: | ---: | --- |
| 退票组1 | 测/试 | MU5231，PKX → PVG(T1) | CNY 0 | CNY 0 | CNY 500 |

请确认是否按以上最终审核结果继续退票。确认后我会执行退票，操作不可撤销；如之后仍需出行，只能重新购票。若不接受这个结果，我会取消这项退票申请。

### 用户

退吧

### 记录

18:36

已处理 17s

### 助手

已确认退票，申请 FRL728N33F 现在是 退票处理中。
本次确认的退款金额：CNY 500
退票费：CNY 0
服务费：CNY 0
航段：MU5231，PKX → PVG(T1)
退款还在处理中，暂时没有明确到账时间。你可以随时让我再查进度。

### 记录

18:36

### 用户

啥时候退票成功，要多久

### 记录

18:37

### 助手

现在状态是 退票处理中，还没显示"退款完成"。
系统里没有给出明确到账时间，所以我不能承诺具体几分钟或几天。一般这种已经确认后的退票，会继续等供应商/财务侧处理；你可以稍后让我查申请 FRL728N33F 的进度。

### 记录

18:37

### 用户

现在什么情况

### 记录

18:39

已处理 20s

### 助手

现在退票已经执行成功，处于 已退票，退款处理中。
申请编号：FRL728N33F
原订单：FOL728823L
实际退款：CNY 500
退票费 / 服务费：CNY 0 / CNY 0
票号：TEST01 已标记为已退票。

也就是说票已经退掉了，现在只等退款入账完成。

### 用户

啥时候入账？

### 助手

目前系统只显示 退款处理中，没有给出具体入账时间，所以我不能保证几点或哪天到账。

已确认的信息是：退款金额 CNY 500，退票费和服务费都是 CNY 0。
你可以晚点让我再查 FRL728N33F，只要状态变成"退款完成"，就说明入账完成了。

### 用户

再查一下

### 助手

已经入账完成了。

退票申请 FRL728N33F 状态现在是 已退票，退款完成。
退款金额：CNY 500，退票费和服务费都是 CNY 0。

## 余额查询

### 用户

查一下我的余额

### 助手

当前余额：CNY -810
授信额度：CNY 2,000
账户状态正常，可继续使用。

## 查询是否还有行程

### 用户

查一下我的订单，是否还有行程？

### 助手

有，你现在还有一段有效行程：

订单 FOL728WX9H
航班 MF8150：2026-08-31 13:15 PKX → 15:35 HGH(T3)
状态：已出票
票号：TEST00

刚刚那张北京到上海的订单 FOL728823L 已经退票并退款完成，所以那段上海行程已经没有有效票了。

## 退北京到杭州订单

### 用户

这张也帮我退了

### 助手

这张也可以退。最新查询显示乘客 测/试 可提交退票申请。

退票范围：

订单：FOL728WX9H
乘客：测/试
航段：MF8150，2026-08-31 13:15 PKX → 15:35 HGH(T3)
票号：TEST00
退票原因：自愿退票

确认后我会先提交退票申请，申请需要供应商审核，不会立即退票。审核完成后，我会告诉你最终可以退回多少钱，再由你决定是否继续执行退票。

如果确认提交，请回复：确认提交退票申请。

### 用户

确认

### 助手

退票申请已提交。

申请编号：FRL728VV6Q
原订单号：FOL728WX9H
乘客：测/试
航段：MF8150，2026-08-31 13:15 PKX → 15:35 HGH(T3)
原因：自愿退票

退票申请正在审核中，请耐心等待。审核完成后，我再查最终可以退回多少钱，然后由你决定是否继续执行退票。
"""

BLOCK_STRUCTURED = """乘客1：
姓：测
名：试
性别：男
出生日期：1998-01-01
国籍：CN
证件类型：身份证
证件号：110235199801013127

联系人：
联系人姓名：测试
邮箱：test@example.com
电话区号：86
手机号：13800000000"""

BLOCK_NATURAL = """乘客就写测试吧，姓测名试，男的，98年1月1号出生的，中国国籍，
用身份证，号码是110235199801013127。联系人也填他本人，
邮箱是test@example.com，手机号13800000000，区号86。"""

# ────────────────────────────── 目标与校验 ──────────────────────────────

FIELDS = ["surname", "given_name", "gender", "birth_date", "nationality",
          "id_type", "id_no", "contact_name", "contact_email", "phone_code", "phone"]

GROUND = {
    "surname": "测", "given_name": "试", "gender": "男", "birth_date": "1998-01-01",
    "nationality": "CN", "id_type": "身份证", "id_no": "110235199801013127",
    "contact_name": "测试", "contact_email": "test@example.com",
    "phone_code": "86", "phone": "13800000000",
}

ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
ID_CHECKMAP = "10X98765432"


def id_checksum_ok(s: str) -> bool:
    if not re.fullmatch(r"\d{17}[\dXx]", s or ""):
        return False
    total = sum(int(s[i]) * ID_WEIGHTS[i] for i in range(17))
    return ID_CHECKMAP[total % 11] == s[17].upper()


def field_valid(f: str, v) -> bool:
    if not isinstance(v, str) or not v.strip():
        return False
    v = v.strip()
    if f == "birth_date":
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return True
        except Exception:
            return False
    if f == "phone":
        return bool(re.fullmatch(r"1[3-9]\d{9}", v))
    if f == "phone_code":
        return bool(re.fullmatch(r"\d{1,3}", v))
    if f == "contact_email":
        return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v))
    if f == "gender":
        return v in ("男", "女")
    if f == "nationality":
        return bool(re.fullmatch(r"[A-Z]{2}", v))
    if f == "id_no":
        return bool(re.fullmatch(r"\d{17}[\dXx]", v))
    return True


def norm(v):
    return re.sub(r"\s+", "", str(v)).strip().lower() if v is not None else ""


# ────────────────────────────── 输入形态 ──────────────────────────────

PII_KEYWORDS = ("姓", "名", "性别", "出生", "国籍", "证件", "身份证", "护照",
                "邮箱", "手机", "电话", "区号", "@", "联系人")


def make_slice(text: str) -> str:
    """规则切片：只保留含 PII 关键词的行 + 相邻 1 行上下文"""
    lines = text.split("\n")
    keep = set()
    for i, ln in enumerate(lines):
        if any(k in ln for k in PII_KEYWORDS):
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(lines):
                    keep.add(j)
    return "\n".join(lines[i] for i in sorted(keep))


VARIANTS = {
    "A 全量原文(结构化)": TRANSCRIPT.replace("@@PASSENGER_BLOCK@@", BLOCK_STRUCTURED),
    "B 规则切片(结构化)": make_slice(TRANSCRIPT.replace("@@PASSENGER_BLOCK@@", BLOCK_STRUCTURED)),
    "C 全量原文(自然语言)": TRANSCRIPT.replace("@@PASSENGER_BLOCK@@", BLOCK_NATURAL),
}

# ────────────────────────────── 正则基线 ──────────────────────────────

REGEX_PATS = {
    "surname": r"姓[：:]\s*(\S+)",
    "given_name": r"名[：:]\s*(\S+)",
    "gender": r"性别[：:]\s*(男|女)",
    "birth_date": r"出生日期[：:]\s*(\d{4}-\d{2}-\d{2})",
    "nationality": r"国籍[：:]\s*([A-Za-z]{2})",
    "id_type": r"证件类型[：:]\s*(\S+)",
    "id_no": r"证件号[：:]?\s*(\d{17}[\dXx])",
    "contact_name": r"联系人姓名[：:]\s*(\S+)",
    "contact_email": r"邮箱[：:]\s*(\S+@\S+)",
    "phone_code": r"电话区号[：:]\s*(\d{1,3})",
    "phone": r"手机号[：:]\s*(1[3-9]\d{9})",
}


def regex_extract(text: str):
    out = {}
    for f, pat in REGEX_PATS.items():
        m = re.search(pat, text)
        out[f] = m.group(1) if m else None
    return out


# ────────────────────────────── 模型抽取 ──────────────────────────────

SYS_BASE = """你是信息抽取器。
规则：
1. 禁止输出思考过程，禁止标签，禁止 markdown ``` 标记。
2. 只输出裸 JSON，不输出其他任何文字。
3. 从【对话记录】里找出乘机人和联系人的资料，逐字填入下面字段。
4. 找不到就填 null，不要编造、不要推测、不要翻译、不要改写原文。
5. 出生日期统一转成 YYYY-MM-DD 格式；国籍用两字母国家代码。"""

SPEC = """
字段结构：
{"surname":"姓","given_name":"名","gender":"男或女","birth_date":"YYYY-MM-DD",
"nationality":"两字母国家代码","id_type":"证件类型","id_no":"证件号码",
"contact_name":"联系人姓名","contact_email":"邮箱","phone_code":"区号数字","phone":"手机号数字"}"""

FEWSHOT = """
示例：
对话记录：乘客1：姓：李 名：雷 性别：女 出生日期：1990-05-20 国籍：CN 证件类型：护照 证件号：E12345678
联系人：联系人姓名：李雷 邮箱：test@example.com 电话区号：86 手机号：13800000000
输出：{"surname":"李","given_name":"雷","gender":"女","birth_date":"1990-05-20","nationality":"CN","id_type":"护照","id_no":"E12345678","contact_name":"李雷","contact_email":"test@example.com","phone_code":"86","phone":"13800000000"}
"""


def build_prompt(tok, transcript: str, fewshot: bool):
    sys_p = SYS_BASE + SPEC + (FEWSHOT if fewshot else "")
    user = f"【对话记录】\n{transcript}\n\n【输出】\n"
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```", 2)
        s = parts[1] if len(parts) > 1 else s
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        s = s[a:b + 1]
    return s


def llm_extract(model, tok, transcript: str, fewshot: bool, max_tokens=320):
    p = build_prompt(tok, transcript, fewshot)
    t0 = time.time()
    raw = model if False else None
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    raw = generate(model, tok, prompt=p, max_tokens=max_tokens,
                   sampler=make_sampler(temp=0.0), verbose=False)
    dt = time.time() - t0
    s = strip_fence(raw)
    try:
        v = json.loads(s)
        if not isinstance(v, dict):
            v = None
    except Exception:
        v = None
    return v, raw, dt, len(p)


# ────────────────────────────── 评分 ──────────────────────────────

def score(result: dict):
    """返回 (字段正确数, 字段有效数, 缺字段数, 错值清单)"""
    correct = valid = 0
    wrong = []
    for f in FIELDS:
        v = result.get(f) if isinstance(result, dict) else None
        ok_valid = field_valid(f, v)
        ok_exact = ok_valid and norm(v) == norm(GROUND[f])
        valid += ok_valid
        correct += ok_exact
        if not ok_exact:
            wrong.append((f, v, ok_valid))
    return correct, valid, wrong


def report(tag, result, raw, dt=None, plen=None):
    correct, valid, wrong = score(result)
    ok_json = isinstance(result, dict)
    flag = ""
    if not ok_json:
        flag = "  <<JSON 解析失败"
    print(f"\n--- {tag} ---")
    print(f"    字段抽取正确 {correct}/{len(FIELDS)}   格式校验通过 {valid}/{len(FIELDS)}{flag}")
    if dt is not None:
        print(f"    延迟 {dt:.2f}s   prompt 长度 {plen} 字符")
    if ok_json:
        missing = [f for f in FIELDS if not result.get(f)]
        if missing:
            print(f"    缺失字段: {missing}")
    if wrong:
        detail = "; ".join(
            f"{f}={repr(v)[:24]}{'(格式非法)' if not okv else ''}"
            for f, v, okv in wrong[:6])
        print(f"    错误明细: {detail}")
    if not ok_json and raw:
        print(f"    raw: {raw[:160]!r}")


def main():
    print("== 探针 v8：350M 做长对话用户信息抽取 ==")
    print(f"目标字段 {len(FIELDS)} 个；ground truth 来自语料原文\n")

    print("== 各形态输入规模 ==")
    for k, v in VARIANTS.items():
        print(f"  {k:<22} {len(v):>6} 字符")

    # 1) 正则基线（纯代码，零 LLM）
    print("\n\n########## 对照组：纯正则基线（零模型调用） ##########")
    for name, text in VARIANTS.items():
        report(f"regex × {name}", regex_extract(text), None)

    # 2) 350M
    print("\n\n########## 实验组：LFM2.5-350M-MLX-4bit ##########")
    from mlx_lm import load
    t0 = time.time()
    model, tok = load("models/LFM2.5-350M-MLX-4bit",
                      tokenizer_config={"trust_remote_code": True})
    print(f"模型加载 {time.time() - t0:.1f}s\n")

    for name, text in VARIANTS.items():
        for few in (False, True):
            res, raw, dt, plen = llm_extract(model, tok, text, few)
            report(f"350M × {name} × {'few-shot' if few else '0-shot'}",
                   res, raw, dt, plen)

    # 3) 校验器能力演示
    print("\n\n########## 程序校验器能力 ##########")
    gt_id = GROUND["id_no"]
    print(f"  证件号 {gt_id}  mod11-2 校验位: {'通过' if id_checksum_ok(gt_id) else '不通过'}")
    print("  （该语料是测试账号，身份证号为合成数据，校验位本就不通过 ——")
    print("    这恰好演示了校验器能在字段'抽对了'的前提下仍抓出数据问题，触发升级/人工。）")


if __name__ == "__main__":
    main()
