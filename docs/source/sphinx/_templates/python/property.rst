{% if obj.display %}
   {% if is_own_page %}
.. _{{ obj.id }}:

.. rst-class:: py-property

:octicon:`cache` {{ obj.short_name }}
{{ "=" * (17 + obj.short_name | length) }}

   {% endif %}

{% if current_module %}
.. currentmodule:: {{ current_module }}
{% endif %}

.. py:property:: {% if not containing_class %}{{ obj.id }}{% else %}{{ containing_class }}.{{ obj.name }}{% endif %}
   {% if obj.annotation %}

   :type: {{ obj.annotation }}
   {% endif %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
