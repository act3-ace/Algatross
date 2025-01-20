{% if obj.display %}
   {% if is_own_page %}
.. _{{ obj.id }}:

.. rst-class:: py-method

:octicon:`cpu` {{ obj.short_name }}
{{ "=" * (15 + obj.short_name | length) }}

   {% endif %}

{% if current_module %}
.. currentmodule:: {{ current_module }}
{% endif %}

.. py:method:: {% if not containing_class %}{{ obj.id }}{% else %}{{ containing_class }}.{{ obj.name }}{% endif %}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}
   {% for (args, return_annotation) in obj.overloads %}

               {%+ if not containing_class %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
