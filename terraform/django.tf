resource "aws_route53_zone" "this" {
  name = "mixtape.kitware.com"
}

output "dns_nameservers" {
  value = aws_route53_zone.this.name_servers
}
