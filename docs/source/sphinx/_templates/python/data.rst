{% if obj.display %}
   {% if is_own_page %}
.. _{{ obj.id }}:

.. rst-class:: py-{{ obj.type }}

:octicon:`database` {{ obj.short_name }}
{{ "=" * (20 + obj.short_name | length) }}

   {% endif %}

{% if current_module %}
.. currentmodule:: {{ current_module }}
{% endif %}

.. py:{{ obj.type }}:: {% if not containing_class %}{{ obj.id }}{% else %}{{ containing_class }}.{{ obj.name }}{% endif %}
   {% if obj.annotation is not none %}

   :type: {% if obj.annotation %} {{ obj.annotation }}{% endif %}
   {% endif %}
   {% if obj.value is not none %}

      {% if obj.value.splitlines()|count > 1 %}
   :value: Multiline-String

   .. raw:: html

      <details><summary>Show Value</summary>

   .. code-block:: python

      {{ obj.value|indent(width=6,blank=true) }}

   .. raw:: html

      </details>

      {% else %}
   :value: {{ obj.value|truncate(100) }}
      {% endif %}
   {% endif %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
