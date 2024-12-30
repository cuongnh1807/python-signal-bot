import re
from collections import defaultdict


def parse_expression(expr):
    """Phân tích một biểu thức thành các thành phần"""
    # Loại bỏ dấu ngoặc và khoảng trắng
    expr = expr.strip('()').strip()

    terms = []
    # Tách các số hạng bằng dấu +
    parts = re.findall(r'([+-]?\d*[a-z]\^?-?\d*|\d+)', expr)

    for part in parts:
        if part:
            coeff = 1
            var = ''
            power = 1

            # Xử lý hệ số
            match = re.match(r'([+-]?\d*)', part)
            if match and match.group(1) and match.group(1) not in ['+', '-']:
                coeff = int(match.group(1))
            elif match and match.group(1) == '-':
                coeff = -1

            # Xử lý biến và số mũ
            var_match = re.search(r'([a-z])', part)
            if var_match:
                var = var_match.group(1)
                power_match = re.search(r'\^(-?\d+)', part)
                if power_match:
                    power = int(power_match.group(1))
            else:
                var = ''
                power = 0

            terms.append((coeff, var, power))

    return terms


def multiply_terms(term1, term2):
    """Nhân hai số hạng"""
    coeff1, var1, power1 = term1
    coeff2, var2, power2 = term2

    # Nhân hệ số
    new_coeff = coeff1 * coeff2

    # Xử lý biến và số mũ
    if var1 == var2:
        new_var = var1
        new_power = power1 + power2
    elif not var1:
        new_var = var2
        new_power = power2
    elif not var2:
        new_var = var1
        new_power = power1
    else:
        new_var = var1 + var2
        new_power = power1 + power2

    return (new_coeff, new_var, new_power)


def multiply_expressions(expr1, expr2):
    """Nhân hai biểu thức"""
    terms1 = parse_expression(expr1)
    terms2 = parse_expression(expr2)

    result = defaultdict(int)

    for term1 in terms1:
        for term2 in terms2:
            new_term = multiply_terms(term1, term2)
            key = (new_term[1], new_term[2])  # key là (biến, số mũ)
            result[key] += new_term[0]  # cộng dồn hệ số

    return result


def format_result(result):
    """Format kết quả thành chuỗi"""
    terms = []

    # Sắp xếp theo số mũ giảm dần
    for (var, power), coeff in sorted(result.items(), key=lambda x: (-x[0][1] if x[0][1] is not None else 0)):
        if coeff == 0:
            continue

        term = ''
        if coeff != 1 or not var:
            if coeff == -1 and var:
                term += '-'
            else:
                term += str(coeff)

        if var:
            term += var
            if power != 1:
                term += f'^{power}'

        terms.append(term)

    return ' + '.join(terms).replace(' + -', ' - ').replace(' ', '')


def evaluate_polynomial(input_str):
    """Hàm chính để xử lý đầu vào"""
    # Tách các biểu thức trong ngoặc
    expressions = re.findall(r'\((.*?)\)', input_str)

    if len(expressions) < 2:
        return "Cần ít nhất 2 biểu thức trong ngoặc"

    # Bắt đầu với hai biểu thức đầu tiên
    result = multiply_expressions(expressions[0], expressions[1])

    # Nhân với các biểu thức còn lại (nếu có)
    for expr in expressions[2:]:
        result = multiply_expressions(format_result(result), expr)

    return format_result(result)


# Test
input_str = "(2x^2+4)(6x^3+3)"
result = evaluate_polynomial(input_str)
print(result)  # Kết quả: x + 2x^-1
