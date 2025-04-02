pipeline {
    agent any

    environment{
        registry = 'khoatomato/primary-agent'
        registryCredential = 'dockerhub'
        imageTag = "v1.$BUILD_NUMBER"
    }

    stages {
        stage('Build and Push') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    def dockerImage = docker.build("${registry}:${imageTag}", "-f ./Primary_agent/Dockerfile ./Primary_agent")
                    echo 'Pushing image to dockerhub..'
                    docker.withRegistry( '', registryCredential ) {
                        dockerImage.push()
                    }
                }
            }
        }

        stage('Deploy') {
            agent {
                kubernetes {
                    containerTemplate {
                        name 'helm' // Name of the container to be used for helm upgrade
                        image 'khoatomato/jenkins-k8s:v1' // The image containing helm
                        alwaysPullImage true // Always pull image in case of using the same tag
                    }
                }
            }
            steps {
                script {
                    container('helm') {
                        sh("helm upgrade --install primary-agent ./Primary_agent/helm_primary_agent/ --namespace primary-agent --set deployment.image.name=${registry} --set deployment.image.tag=${imageTag}")
                    }
                }
            }
        }
    }
}