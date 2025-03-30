# Variables to use across the project

variable "project_id" {
  description = "The project ID to host the cluster in"
  type        = string
  default     = "llmops-455207"
}

variable "region" {
  description = "The region in which the cluster will be deployed"
  type        = string
  default     = "asia-southeast1"
}

variable "zone" {
  description = "The zone within the region where the cluster will be deployed"
  type        = string
  default     = "asia-southeast1-c"
}

variable "self_link" {
  description = "The self_link of the network to attach this firewall to"
  type        = string
  default     = "global/networks/default"
}
