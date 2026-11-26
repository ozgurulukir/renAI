"""Config-layer user-facing messages (folder rules, template validation)."""


class Warn:
    @staticmethod
    def folder_rule_template_ignored(rule: str, key: str, template: str, reason: str) -> str:
        """Warning for a folder-rule template that failed validation.

        Names the rule, the template and the reason it was rejected, so users
        can fix the template without losing the rest of the rule (categories
        are kept).
        """
        return f"Folder rule {rule}: ignoring {key} template {template!r}: {reason}"
