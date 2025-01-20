{% if obj.display %}
   {% if is_own_page %}
.. _{{ obj.id }}:

.. rst-class:: py-function

:octicon:`cpu` {{ obj.short_name }}
{{ "=" * (15 + obj.short_name | length) }}

   {% endif %}
   {% if not current_module %}{% set current_module = obj.id.split(".")[:-1]|join(".") %}{% endif %}
{% if current_module %}
.. currentmodule:: {{ current_module }}
{% endif %}

.. py:function:: {{ obj.short_name }}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}
   {% for (args, return_annotation) in obj.overloads %}

                 {{ obj.short_name }}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
