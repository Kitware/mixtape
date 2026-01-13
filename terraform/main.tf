terraform {
  required_version = ">= 1.1"

  cloud {
    organization = "kitware"

    workspaces {
      name = "mixtape"
    }
  }

  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  # Must set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY envvars
}
