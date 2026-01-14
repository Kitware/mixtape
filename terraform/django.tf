resource "aws_route53_zone" "this" {
  name = "mixtape.kitware.com"
}

data "heroku_team" "this" {
  name = "kitware"
}

module "django" {
  source  = "kitware-resonant/resonant/heroku"
  version = "3.1.0"

  project_slug           = "kw-mixtape"
  route53_zone_id        = aws_route53_zone.this.zone_id
  heroku_team_name       = data.heroku_team.this.name
  subdomain_name         = "www"
  django_settings_module = "mixtape.settings.heroku_production"

  additional_django_vars = {
    DJANGO_SENTRY_DSN = "https://2dce9ed0d8fbcb80869cf8af1c4d36b9@o267860.ingest.us.sentry.io/4510144581074944"
  }
}

output "dns_nameservers" {
  value = aws_route53_zone.this.name_servers
}
