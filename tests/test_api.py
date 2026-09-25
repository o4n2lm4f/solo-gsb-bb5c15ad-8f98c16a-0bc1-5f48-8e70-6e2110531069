"""HTTP API 测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_balance_success():
    resp = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["balanced"] is True
    vector = data["coefficients"]["vector"]
    assert vector == [2, 1, 2]
    assert data["normalized"]["equation"] == "2H2 + O2 -> 2H2O"
    assert data["charge_balance"]["balanced"] is True
    assert data["cached"] is False


def test_idempotent_same_normalized_reaction():
    payload_a = {"reactants": ["Ca(OH)2", "HCl"], "products": ["CaCl2", "H2O"]}
    # 等价写法：方括号替代圆括号、内部空白不同
    payload_b = {
        "reactants": ["Ca [OH]2", " HCl"],
        "products": ["CaCl2", "H2 O"],
    }

    r1 = client.post("/api/v1/balance", json=payload_a)
    r2 = client.post("/api/v1/balance", json=payload_b)
    assert r1.status_code == 200 and r2.status_code == 200
    d1, d2 = r1.json(), r2.json()
    assert d1["reaction_id"] == d2["reaction_id"]
    assert d2["cached"] is True
    assert d1["coefficients"]["vector"] == d2["coefficients"]["vector"]


def test_idempotent_order_independent():
    r1 = client.post(
        "/api/v1/balance",
        json={"reactants": ["O2", "H2"], "products": ["H2O"]},
    )
    r2 = client.post(
        "/api/v1/balance",
        json={"reactants": ["H2", "O2"], "products": ["H2O"]},
    )
    assert r1.json()["reaction_id"] == r2.json()["reaction_id"]


def test_idempotent_equivalent_element_order_h2o_oh2(monkeypatch):
    """组成与电荷相同、仅元素排列不同（H2O vs OH2）必须命中同一 reaction_id。

    命中后按本次输入回显化学式，系数等计算结果复用缓存；/analyze 同理。
    用独立内存库隔离，避免依赖其它测试的缓存状态。
    """
    from app import service
    from app.cache import ReactionCache

    monkeypatch.setattr(service, "cache", ReactionCache(":memory:"))

    first = client.post(
        "/api/v1/balance",
        json={"reactants": ["H2", "O2"], "products": ["H2O"]},
    )
    second = client.post(
        "/api/v1/balance",
        json={"reactants": ["H2", "O2"], "products": ["OH2"]},
    )
    assert first.status_code == second.status_code == 200
    d1, d2 = first.json(), second.json()
    assert d1["reaction_id"] == d2["reaction_id"]
    assert d1["cached"] is False and d2["cached"] is True
    # 计算结果复用缓存
    assert d1["coefficients"]["vector"] == d2["coefficients"]["vector"] == [2, 1, 2]
    assert d2["charge_balance"]["balanced"] is True
    # 展示按本次输入
    assert d2["normalized"]["products"] == ["OH2"]
    assert d2["normalized"]["equation"] == "2H2 + O2 -> 2OH2"
    prod = d2["coefficients"]["products"][0]
    assert prod["input"] == "OH2"
    assert prod["normalized"] == "OH2"
    assert prod["coefficient"] == 2

    # /analyze 使用同一签名并命中
    analyzed = client.post(
        "/api/v1/analyze",
        json={"reactants": ["H2", "O2"], "products": ["OH2"]},
    )
    assert analyzed.status_code == 200
    a = analyzed.json()
    assert a["reaction_id"] == d1["reaction_id"]
    assert a["cached"] is True
    assert a["normalized"]["products"] == ["OH2"]
    assert a["normalized"]["equation"] == "2H2 + O2 -> 2OH2"
    assert a["balanced"]["species"]["products"][0]["normalized"] == "OH2"
    assert a["balanced"]["species"]["products"][0]["coefficient"] == 2

    # GET 取回时以缓存记录的规范化形式展示
    fetched = client.get(f"/api/v1/reactions/{d1['reaction_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["normalized"]["products"] == ["H2O"]
    assert fetched.json()["coefficients"]["vector"] == [2, 1, 2]


def test_idempotent_side_swap_is_different_id(monkeypatch):
    """调换反应物与生成物侧别应得到不同 reaction_id（侧别保留在签名中）。"""
    from app import service
    from app.cache import ReactionCache

    monkeypatch.setattr(service, "cache", ReactionCache(":memory:"))
    forward = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    ).json()
    swapped = client.post(
        "/api/v1/balance", json={"reactants": ["H2O"], "products": ["H2", "O2"]}
    )
    # 反向反应无法配平（这里只验证签名不同 -> 不是缓存命中的同一 ID）
    assert swapped.json()["reaction_id"] != forward["reaction_id"]


def test_get_reaction_by_id():
    created = client.post(
        "/api/v1/balance", json={"reactants": ["Fe", "O2"], "products": ["Fe2O3"]}
    ).json()
    rid = created["reaction_id"]
    resp = client.get(f"/api/v1/reactions/{rid}")
    assert resp.status_code == 200
    assert resp.json()["reaction_id"] == rid
    assert resp.json()["cached"] is True


def test_get_reaction_not_found():
    resp = client.get("/api/v1/reactions/rxn_doesnotexist")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_balance_formula_parse_error():
    resp = client.post(
        "/api/v1/balance", json={"reactants": ["Xx"], "products": ["H2O"]}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "FORMULA_PARSE_ERROR"
    assert "reaction_id" in body


def test_balance_non_unique_error():
    resp = client.post(
        "/api/v1/balance", json={"reactants": ["C", "O2"], "products": ["CO", "CO2"]}
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "NON_UNIQUE_BALANCE"
    assert body["error"]["details"]["nullity"] == 2


def test_balance_impossible_error():
    resp = client.post(
        "/api/v1/balance", json={"reactants": ["H2O"], "products": ["Au"]}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "IMPOSSIBLE_BALANCE"


def test_duplicate_species_conflict():
    resp = client.post(
        "/api/v1/balance", json={"reactants": ["H2", "H2"], "products": ["H2O"]}
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "DUPLICATE_SPECIES"


def test_parse_error_and_duplicate_species_cached_separately():
    """失败签名必须区分语法差异：先提交含内部空格的解析失败请求，
    再提交同侧重复物种请求，后者不得误命中前者的 400 记录。"""
    parse_error_payload = {"reactants": ["H 2", "H2"], "products": ["H2"]}
    duplicate_payload = {"reactants": ["H2", "H2"], "products": ["H2"]}

    r1 = client.post("/api/v1/balance", json=parse_error_payload)
    assert r1.status_code == 400
    d1 = r1.json()
    assert d1["error"]["code"] == "FORMULA_PARSE_ERROR"
    assert d1["cached"] is False

    # 关键回归断言：同侧重复物种应报 409，而不是命中上面的 400 记录
    r2 = client.post("/api/v1/balance", json=duplicate_payload)
    assert r2.status_code == 409
    d2 = r2.json()
    assert d2["error"]["code"] == "DUPLICATE_SPECIES"
    assert d2["error"]["details"]["species"] == "H2"
    assert d2["cached"] is False
    assert d2["reaction_id"] != d1["reaction_id"]

    # 完全相同的失败请求重复提交：各自复用同一 reaction_id 与错误明细
    r1b = client.post("/api/v1/balance", json=parse_error_payload)
    assert r1b.status_code == 400
    assert r1b.json()["reaction_id"] == d1["reaction_id"]
    assert r1b.json()["cached"] is True
    assert r1b.json()["error"] == d1["error"]

    r2b = client.post("/api/v1/balance", json=duplicate_payload)
    assert r2b.status_code == 409
    assert r2b.json()["reaction_id"] == d2["reaction_id"]
    assert r2b.json()["cached"] is True
    assert r2b.json()["error"] == d2["error"]

    # GET 查询：两条错误记录各自保持各自的结果与状态码
    g1 = client.get(f"/api/v1/reactions/{d1['reaction_id']}")
    assert g1.status_code == 400
    assert g1.json()["error"]["code"] == "FORMULA_PARSE_ERROR"
    assert g1.json()["reaction_id"] == d1["reaction_id"]
    g2 = client.get(f"/api/v1/reactions/{d2['reaction_id']}")
    assert g2.status_code == 409
    assert g2.json()["error"]["code"] == "DUPLICATE_SPECIES"
    assert g2.json()["reaction_id"] == d2["reaction_id"]


def test_analyze_parse_error_and_duplicate_species_cached_separately():
    """逐元素分析接口与配平接口共享同一套失败签名，同样不得串记录。"""
    r1 = client.post(
        "/api/v1/analyze", json={"reactants": ["N 2", "N2"], "products": ["N2"]}
    )
    assert r1.status_code == 400
    assert r1.json()["error"]["code"] == "FORMULA_PARSE_ERROR"

    r2 = client.post(
        "/api/v1/analyze", json={"reactants": ["N2", "N2"], "products": ["N2"]}
    )
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "DUPLICATE_SPECIES"
    assert r2.json()["reaction_id"] != r1.json()["reaction_id"]


def test_parse_error_signature_preserves_order_and_details():
    """失败签名保留提交顺序：先报哪个非法物种取决于顺序，不能互相命中。"""
    r1 = client.post(
        "/api/v1/balance", json={"reactants": ["Xx", "Zz"], "products": ["H2O"]}
    )
    r2 = client.post(
        "/api/v1/balance", json={"reactants": ["Zz", "Xx"], "products": ["H2O"]}
    )
    assert r1.status_code == 400 and r2.status_code == 400
    d1, d2 = r1.json(), r2.json()
    assert d1["reaction_id"] != d2["reaction_id"]
    assert "'Xx'" in d1["error"]["message"]
    assert "'Zz'" in d2["error"]["message"]


def test_duplicate_error_signature_preserves_order_and_details():
    """同样的多重集、不同的提交顺序：报出的重复物种不同，记录各自独立。"""
    r1 = client.post(
        "/api/v1/balance",
        json={"reactants": ["H2", "O2", "O2", "H2"], "products": ["H2O"]},
    )
    r2 = client.post(
        "/api/v1/balance",
        json={"reactants": ["H2", "H2", "O2", "O2"], "products": ["H2O"]},
    )
    assert r1.status_code == 409 and r2.status_code == 409
    d1, d2 = r1.json(), r2.json()
    assert d1["reaction_id"] != d2["reaction_id"]
    assert d1["error"]["details"]["species"] == "O2"
    assert d2["error"]["details"]["species"] == "H2"


def test_invalid_request_body():
    resp = client.post(
        "/api/v1/balance", json={"reactants": [], "products": ["H2O"]}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"

    resp = client.post("/api/v1/balance", json={"reactants": ["H2"]})
    assert resp.status_code == 422


def test_analyze_shows_raw_and_balanced():
    resp = client.post(
        "/api/v1/analyze", json={"reactants": ["H2", "O2"], "products": ["H2O"]}
    )
    assert resp.status_code == 200
    data = resp.json()
    raw_rows = {row["element"]: row for row in data["raw"]["elements"]}
    assert raw_rows["H"]["reactant_total"] == 2
    assert raw_rows["H"]["product_total"] == 2
    assert raw_rows["H"]["difference"] == 0
    assert raw_rows["O"]["reactant_total"] == 2
    assert raw_rows["O"]["product_total"] == 1
    assert raw_rows["O"]["difference"] == 1
    assert data["raw"]["coefficients_assumed"] == 1
    assert data["balanced_ok"] is True
    balanced_rows = {row["element"]: row for row in data["balanced"]["elements"]}
    assert balanced_rows["O"]["reactant_total"] == 2
    assert balanced_rows["O"]["product_total"] == 2
    assert all(row["balanced"] for row in balanced_rows.values())


def test_analyze_non_unique_returns_error_in_body():
    resp = client.post(
        "/api/v1/analyze",
        json={"reactants": ["C", "O2"], "products": ["CO", "CO2"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["balanced_ok"] is False
    assert data["balanced"] is None
    assert data["error"]["code"] == "NON_UNIQUE_BALANCE"
    # 原始视图仍然给出差异
    assert {row["element"] for row in data["raw"]["elements"]} == {"C", "O"}


def test_analyze_charge_rows():
    resp = client.post(
        "/api/v1/analyze",
        json={"reactants": ["MnO4-", "H+", "e-"], "products": ["Mn2+", "H2O"]},
    )
    data = resp.json()
    assert data["balanced"]["charge"] == {
        "reactant_total": 2,
        "product_total": 2,
        "difference": 0,
        "balanced": True,
    }


def test_redox_full_reaction():
    resp = client.post(
        "/api/v1/balance",
        json={
            "reactants": ["Cu", "NO3-", "H+"],
            "products": ["Cu2+", "NO", "H2O"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    coef = {item["normalized"]: item["coefficient"]
            for item in data["coefficients"]["reactants"]
            + data["coefficients"]["products"]}
    assert coef == {"Cu": 3, "NO3^-": 2, "H^+": 8, "Cu^2+": 3, "NO": 2, "H2O": 4}


# ------------------------------------------------------------ /balance/variants
def test_variants_unique_reaction_single_solution():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2", "O2"],
            "products": ["H2O"],
            "max_coefficient": 30,
            "limit": 100,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["nullity"] == 1
    assert data["total"] == data["count"] == 1
    assert data["truncated"] is False
    item = data["variants"][0]
    assert item["coefficients"]["vector"] == [2, 1, 2]
    assert item["sum"] == 5
    assert item["equation"] == "2H2 + O2 -> 2H2O"
    # 归一化块不携带系数，保持物种清单语义
    assert data["normalized"]["equation"] == "H2 + O2 -> H2O"


def test_variants_underdetermined_sorted_and_truncated():
    body = {
        "reactants": ["C", "O2"],
        "products": ["CO", "CO2"],
        "max_coefficient": 10,
        "limit": 5,
    }
    resp = client.post("/api/v1/balance/variants", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["nullity"] == 2
    assert data["total"] == 15
    assert data["count"] == 5
    assert data["truncated"] is True
    vectors = [v["coefficients"]["vector"] for v in data["variants"]]
    assert vectors == [
        [3, 2, 2, 1],
        [4, 3, 2, 2],
        [5, 3, 4, 1],
        [5, 4, 2, 3],
        [6, 5, 2, 4],
    ]
    sums = [v["sum"] for v in data["variants"]]
    assert sums == sorted(sums)


def test_variants_limit_not_exceeded_sets_truncated_false():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["C", "O2"],
            "products": ["CO", "CO2"],
            "max_coefficient": 4,
            "limit": 100,
        },
    )
    data = resp.json()
    assert data["truncated"] is False
    assert data["count"] == data["total"] == 2
    assert [v["coefficients"]["vector"] for v in data["variants"]] == [
        [3, 2, 2, 1],
        [4, 3, 2, 2],
    ]


def test_variants_no_positive_solution_empty_list():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2", "O2", "N2"],
            "products": ["H2O"],
            "max_coefficient": 10,
            "limit": 10,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["variants"] == []
    assert data["count"] == data["total"] == 0
    assert data["truncated"] is False


def test_variants_space_too_large():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2", "O2", "H2O", "H2O2", "O3"],
            "products": ["HO2"],
            "max_coefficient": 10,
            "limit": 10,
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "VARIANT_SPACE_TOO_LARGE"
    assert body["error"]["details"]["nullity"] == 4
    assert "reaction_id" in body


def test_variants_impossible_balance():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2O"],
            "products": ["Au"],
            "max_coefficient": 10,
            "limit": 10,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "IMPOSSIBLE_BALANCE"


def test_variants_parse_error():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["Xx"],
            "products": ["H2O"],
            "max_coefficient": 5,
            "limit": 10,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "FORMULA_PARSE_ERROR"


def test_variants_duplicate_species():
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2", "H2"],
            "products": ["H2O"],
            "max_coefficient": 5,
            "limit": 10,
        },
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "DUPLICATE_SPECIES"


@pytest.mark.parametrize("payload", [
    {"reactants": ["H2"], "products": ["H2O"], "max_coefficient": 1, "limit": 10},
    {"reactants": ["H2"], "products": ["H2O"], "max_coefficient": 31, "limit": 10},
    {"reactants": ["H2"], "products": ["H2O"], "max_coefficient": 10, "limit": 0},
    {"reactants": ["H2"], "products": ["H2O"], "max_coefficient": 10, "limit": 101},
    {"reactants": ["H2"], "products": ["H2O"], "max_coefficient": "x", "limit": 10},
    {"reactants": [], "products": ["H2O"], "max_coefficient": 10, "limit": 10},
])
def test_variants_invalid_params(payload):
    resp = client.post("/api/v1/balance/variants", json=payload)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_variants_reaction_id_matches_balance_endpoint():
    balance = client.post(
        "/api/v1/balance",
        json={"reactants": ["Ca(OH)2", "HCl"], "products": ["CaCl2", "H2O"]},
    ).json()
    variants = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["Ca [OH]2", " HCl"],
            "products": ["CaCl2", "H2 O"],
            "max_coefficient": 20,
            "limit": 10,
        },
    ).json()
    assert balance["reaction_id"] == variants["reaction_id"]


def test_variants_cross_side_same_species_keeps_separate_coefficients():
    """H2 同时出现在反应两侧：reactants/products 明细与 equation 必须各自
    对应 vector 的位置，跨侧同物种系数不同也不能相互覆盖。"""
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2", "O2"],
            "products": ["H2", "H2O"],
            "max_coefficient": 5,
            "limit": 100,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["normalized"] == {
        "reactants": ["H2", "O2"],
        "products": ["H2", "H2O"],
        "equation": "H2 + O2 -> H2 + H2O",
    }
    assert data["total"] == data["count"] == 4
    assert data["truncated"] is False

    atom_counts = {
        "H2": {"H": 2},
        "O2": {"O": 2},
        "H2O": {"H": 2, "O": 1},
    }
    expected_equations = [
        "3H2 + O2 -> H2 + 2H2O",
        "4H2 + O2 -> 2H2 + 2H2O",
        "5H2 + O2 -> 3H2 + 2H2O",
        "5H2 + 2O2 -> H2 + 4H2O",
    ]
    for item, expected_equation in zip(data["variants"], expected_equations):
        reactants = item["coefficients"]["reactants"]
        products = item["coefficients"]["products"]
        vector = item["coefficients"]["vector"]

        # 明细顺序与规范化后的侧别顺序一致
        assert [x["normalized"] for x in reactants] == ["H2", "O2"]
        assert [x["normalized"] for x in products] == ["H2", "H2O"]

        # 明细系数与 vector 按位置一一对应（反应物 H2 的系数不被生成物 H2 覆盖）
        side_coefficients = [x["coefficient"] for x in reactants] + [
            x["coefficient"] for x in products
        ]
        assert side_coefficients == vector

        # equation 与明细完全一致
        assert item["equation"] == expected_equation

        # 展示方程逐元素守恒
        def side_totals(side):
            totals = {}
            for x in side:
                for element, count in atom_counts[x["normalized"]].items():
                    totals[element] = totals.get(element, 0) + x["coefficient"] * count
            return totals

        assert side_totals(reactants) == side_totals(products)

    # 问题示例：vector [3,1,1,2] 中反应物 H2 必须显示为 3
    first = data["variants"][0]["coefficients"]
    assert first["vector"] == [3, 1, 1, 2]
    assert first["reactants"][0] == {
        "input": "H2",
        "normalized": "H2",
        "coefficient": 3,
    }
    assert first["products"][0] == {
        "input": "H2",
        "normalized": "H2",
        "coefficient": 1,
    }


def test_variants_cross_side_same_species_limit_truncation():
    """跨侧同物种方案同样遵守总数、排序与 limit 截取语义。"""
    resp = client.post(
        "/api/v1/balance/variants",
        json={
            "reactants": ["H2", "O2"],
            "products": ["H2", "H2O"],
            "max_coefficient": 5,
            "limit": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 4
    assert data["count"] == 2
    assert data["truncated"] is True
    assert [v["coefficients"]["vector"] for v in data["variants"]] == [
        [3, 1, 1, 2],
        [4, 1, 2, 2],
    ]
    assert [v["equation"] for v in data["variants"]] == [
        "3H2 + O2 -> H2 + 2H2O",
        "4H2 + O2 -> 2H2 + 2H2O",
    ]


def test_balance_endpoint_unchanged_payload_shape():
    # 回归保护：/balance 的字段与缓存语义不受新接口影响
    resp = client.post(
        "/api/v1/balance",
        json={"reactants": ["H2", "O2"], "products": ["H2O"]},
    )
    data = resp.json()
    assert set(data) == {
        "reaction_id", "cached", "normalized", "coefficients",
        "balanced", "elements", "charge_balance",
    }
    assert data["normalized"]["equation"] == "2H2 + O2 -> 2H2O"
