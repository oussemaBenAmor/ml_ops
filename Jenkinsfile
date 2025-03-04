pipeline {
    agent any  

    environment {
        VENV_DIR = 'venv'  
        MODEL_PATH = "best_svm_model.pkl"
        DOCKER_IMAGE = 'benamoroussema/mlops_app:latest'  // Docker image name
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

        stage('Code Quality Check') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Code Quality Check' }
            }
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    ${VENV_DIR}/bin/black main.py model_pipeline.py
                    ${VENV_DIR}/bin/flake8 --exit-zero main.py model_pipeline.py
                    ${VENV_DIR}/bin/bandit --exit-zero main.py model_pipeline.py
                '''
            }
        }

        stage('Set up Environment') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Set up Environment' }
            }
            steps {
                sh 'python3 -m venv ${VENV_DIR}'
                sh '. ${VENV_DIR}/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Docker Run') {
            when {
                expression { params.RUN_STAGE == 'ALL' || params.RUN_STAGE == 'Docker Run' }
            }
            steps {
                sh '''
                    # Find the process using port 5001
                    PID=$(lsof -t -i :5001 || true)
                    # If a process is found, kill it
                    if [ -n "$PID" ]; then
                        kill -9 $PID
                    fi
                    . ${VENV_DIR}/bin/activate
                    # Ensure the container doesn't already exist
                    docker stop mlops_container || true
                    docker rm mlops_container || true
                    
                    # Now run the Docker container
                    docker run -d -p 5000:5000 -p 5001:5001 --name mlops_container ${DOCKER_IMAGE}
                '''
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
        
       

        
        
    
    }
}

