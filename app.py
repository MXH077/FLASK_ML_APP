from flask import Flask, request, render_template
import joblib
import pandas as pd  # 导入 pandas

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'your_secret_key'

# 加载模型、scaler 和 PCA
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# 特征列名与训练时一致
feature_names = ['id', 'loanAmnt', 'interestRate', 'installment', 'grade', 'subGrade', 
                 'employmentTitle', 'annualIncome', 'isDefault', 'purpose', 'dti', 
                 'ficoRangeLow', 'openAcc', 'revolBal', 'totalAcc', 'title', 'policyCode']

# 定义 grade 映射
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取表单数据
        loanAmnt = float(request.form['loanAmnt'])
        print(f"loanAmnt: {loanAmnt}")  # 获取表单数据并打印调试信息

        installment = float(request.form['installment'])
        print(f"installment: {installment}") 

        subGrade = request.form['subGrade']  # 获取字符串输入
        print(f"subGrade: {subGrade}") 

        interestRate = float(request.form['interestRate'])
        print(f"interestRate: {interestRate}")

        ficoRangeLow = float(request.form['ficoRangeLow'])
        print(f"ficoRangeLow: {ficoRangeLow}")

        revolBal = float(request.form['revolBal'])
        print(f"revolBal: {revolBal}")

        totalAcc = float(request.form['totalAcc'])
        print(f"totalAcc: {totalAcc}")

        dti = float(request.form['dti'])
        print(f"dti: {dti}")

        purpose = int(request.form['purpose'])  # 将purpose转换为整数
        print(f"purpose: {purpose}")

        isDefault = int(request.form['isDefault']) # 将 isDefault 转换为整数
        print(f"isDefault: {isDefault}")

        grade = request.form['grade']
        print(f"grade: {grade}")

        annualIncome = float(request.form['annualIncome'])
        print(f"annualIncome: {annualIncome}")

        employmentTitle = int(request.form['employmentTitle'])
        print(f"employmentTitle: {employmentTitle}")

        id = int(request.form['id'])
        print(f"id: {id}")

        title = int(request.form['title'])
        print(f"title: {title}")  

        policyCode = int(request.form['policyCode'])  # 固定为 1
        print(f"policyCode: {policyCode}")

        openAcc = int(request.form['openAcc'])  # 新增 openAcc 特征
        print(f"openAcc: {openAcc}")

        # 将 subGrade 字符串拆分为字母和数字，并转换为数值
        if len(subGrade) == 2:
            subGrade_letter = ord(subGrade[0]) - ord('A') + 1
            subGrade_number = int(subGrade[1])
            subGrade_numeric = subGrade_letter * 10 + subGrade_number
        else:
            raise ValueError("Invalid subGrade format. It should be in the format 'A1'.")     

        # 处理 `grade` 字段
        if grade not in grade_mapping:
            raise ValueError(f"Invalid grade: {grade}")
        grade_numeric = grade_mapping[grade]

        # 确保所有字段都有值
        if any(v is None for v in [loanAmnt, interestRate, installment, subGrade_numeric, ficoRangeLow,
                                    revolBal, totalAcc, dti, purpose, isDefault, grade_numeric, annualIncome,
                                    employmentTitle, id, title, policyCode, openAcc]):
            raise ValueError("Some required fields are missing.")
        
        # 创建要输入模型的 DataFrame
        input_data = pd.DataFrame([[id, loanAmnt, interestRate, installment, grade_numeric, subGrade_numeric,
                                    employmentTitle, annualIncome, isDefault, purpose, dti,
                                    ficoRangeLow, openAcc, revolBal, totalAcc, title, policyCode]],
                                  columns=feature_names)


        # 使用模型进行预测
        input_data_scaled = scaler.transform(input_data)  # 对输入数据进行标准化
        prediction = model.predict(input_data_scaled)

        # 由于 prediction 是一个数组或列表，提取其第一个元素
        prediction_result = prediction[0]

        # 将预测结果返回给前端
        return render_template('result.html', prediction=prediction_result)
    
    
    except ValueError as ve:
        return str(ve), 400        # 返回自定义错误信息
    except Exception as e:
        return f"An error occurred: {str(e)}", 500  # 返回通用错误信息

if __name__ == '__main__':
    app.run(debug=True)