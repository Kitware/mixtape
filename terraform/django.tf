data "aws_route53_zone" "this" {
  # This must be created by hand in the AWS console
  name = "mixtape.test"
}

data "heroku_team" "this" {
  name = "kitware"
}

module "django" {
  source  = "kitware-resonant/resonant/heroku"
  version = "1.1.1"

  project_slug     = "mixtape"
  route53_zone_id  = data.aws_route53_zone.this.zone_id
  heroku_team_name = data.heroku_team.this.name
  subdomain_name   = "www"
}
