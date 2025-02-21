pipeline {
    agent any  

    environment {
        VENV_DIR = 'venv'  
        MODEL_PATH = "${WORKSPACE}/best_svm_model.pkl"
        MLFLOW_TRACKING_URI = "http://localhost:5001"
    }

    parameters {
        string(name: 'RUN_STAGE', defaultValue: 'ALL', description: 'Enter stage name to run a single stage or ALL to run everything')
    }

    stages {
        stage('Checkout Code') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Checkout Code' }
            }
            steps {
                git branch: 'main', url:'https://github.com/oussemaBenAmor/ml_ops.git' 
            }
        }

        stage('Set up Environment') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Set up Environment' }
            }
            steps {
                sh 'python3 -m venv ${VENV_DIR}'
                sh '. ${VENV_DIR}/bin/activate && pip install -r requirements.txt'
                sh 'chmod -R 777 ${WORKSPACE}'  // Add this line
            }
        }

        stage('Start MLflow Server') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Start MLflow Server' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && mlflow ui --host 0.0.0.0 --port 5001 &'
            }
        }

        stage('Prepare Data') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Prepare Data' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --prepare'
            }
        }

        stage('Train Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Train Model' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --train'
            }
        }

        stage('Evaluate Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Evaluate Model' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --evaluate'
            }
        }
        
        stage('Improve Model') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Improve Model' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python main.py --improve'
            }
        }

        stage('Deploy API') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Deploy API' }
            }
            steps {
                sh '. ${VENV_DIR}/bin/activate && python app.py'
            }
        }
    }
}
