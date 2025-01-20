{% if obj.display %}
   {% if is_own_page %}
.. _{{ obj.id }}:

.. rst-class:: py-{{ obj.type }}

{% if obj.type == "class" %}{% set octicon=":octicon:`container`" %}{% else %}{% set octicon=":octicon:`alert`" %}{% endif %}
{{ octicon }} {{ obj.short_name }}
{{ "=" * ((octicon | length) + (obj.short_name | length) + 1) }}

   {% endif %}
   {% set visible_children = obj.children|selectattr("display")|list %}
   {% set own_page_children = visible_children|selectattr("type", "in", own_page_types)|list %}
   {% if not current_module %}{% set current_module = obj.id.split(".")[:-1]|join(".") %}{% endif %}
   {% if is_own_page and own_page_children %}
.. toctree::
   :hidden:

      {% for child in own_page_children %}
   {{ child.include_path }}
      {% endfor %}

   {% endif %}

{% if current_module %}
.. currentmodule:: {{ current_module }}
{% endif %}

.. py:{{ obj.type }}:: {% if containing_class %}{{ containing_class }}.{% endif %}{{ obj.short_name }}{% if obj.args %}({{ obj.args }}){% endif %}
   {% for (args, return_annotation) in obj.overloads %}
      {{ " " * (obj.type | length) }}   {% if containing_class %}{{ containing_class }}.{% endif %}{{ obj.short_name }}{% if args %}({{ args }}){% endif %}

   {% endfor %}

   {% if obj.bases %}
      {% if "show-inheritance" in autoapi_options %}

   Bases: {% for base in obj.bases %}{{ base|link_objs }}{% if not loop.last %}, {% endif %}{% endfor %}
      {% endif %}


      {% if "show-inheritance-diagram" in autoapi_options and obj.bases != ["object"] %}
   .. autoapi-inheritance-diagram:: {{ obj.obj["full_name"] }}
      :parts: 1
         {% if "private-members" in autoapi_options %}
      :private-bases:
         {% endif %}

      {% endif %}
   {% endif %}
   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
   {% if is_own_page and visible_children %}
      {% set visible_attributes = visible_children|selectattr("type", "equalto", "attribute")|list %}
      {% if visible_attributes %}
      {% if "attribute" in own_page_types %}
:octicon:`database` Attributes
------------------------------
      {% else %}
.. rubric:: :octicon:`database` Attributes
   :class: attributes
      {% endif %}

.. autoapisummary::
   :nosignatures:

         {% for obj_item in visible_attributes %}
   {{ obj_item.id }}
         {% endfor %}
      {% endif %}

      {% set visible_properties = visible_children|selectattr("type", "equalto", "property")|list %}
      {% if visible_properties %}
      {% if "property" in own_page_types %}
:octicon:`cache` Properties
---------------------------
      {% else %}
.. rubric:: :octicon:`cache` Properties
   :class: properties
      {% endif %}

.. autoapisummary::
   :nosignatures:

         {% for obj_item in visible_properties %}
   {{ obj_item.id }}
         {% endfor %}
      {% endif %}

      {% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
      {% if visible_exceptions %}
      {% if "exception" in own_page_types %}
:octicon:`alert` Exceptions
---------------------------
      {% else %}
.. rubric:: :octicon:`alert` Exceptions
   :class: exceptions
      {% endif %}

.. autoapisummary::
   :nosignatures:

         {% for obj_item in visible_exceptions %}
   {{ obj_item.id }}
         {% endfor %}
      {% endif %}

      {% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
      {% if visible_classes %}
      {% if "class" in own_page_types %}
:octicon:`container` Classes
----------------------------
      {% else %}
.. rubric:: :octicon:`container` Classes
   :class: classes
      {% endif %}

.. autoapisummary::
   :nosignatures:

         {% for obj_item in visible_classes %}
   {{ obj_item.id }}
         {% endfor %}
      {% endif %}

      {% set visible_methods = visible_children|selectattr("type", "equalto", "method")|list %}
      {% if visible_methods %}
      {% if "method" in own_page_types %}
:octicon:`cpu` Methods
----------------------
      {% else %}
.. rubric:: :octicon:`cpu` Methods
   :class: methods
      {% endif %}

.. autoapisummary::
   :nosignatures:

            {% for obj_item in visible_methods %}
   {{ obj_item.id }}
            {% endfor %}
      {% endif %}

      {% if containing_class %}
      {% set sub_containing_class = ".".join([containing_class, short_name]) %}
      {% else %}
      {% set sub_containing_class = short_name %}
      {% endif %}
      {% if visible_attributes and "attribute" not in own_page_types %}
:octicon:`database` Attributes
------------------------------

         {% for obj_item in visible_attributes %}
{{ obj_item.render(is_own_section = True, containing_class = obj.short_name, current_module = current_module)}}
         {% endfor %}
      {% endif %}
      {% if visible_properties and "property" not in own_page_types %}
:octicon:`cache` Properties
---------------------------

         {% for obj_item in visible_properties %}
{{ obj_item.render(is_own_section = True, containing_class = obj.short_name, current_module = current_module) }}
         {% endfor %}
      {% endif %}
      {% if visible_exceptions and "exception" not in own_page_types %}
:octicon:`alert` Exceptions
---------------------------

         {% for obj_item in visible_exceptions %}
{{ obj_item.render(is_own_section = True, containing_class = obj.short_name, current_module = current_module) }}
         {% endfor %}
      {% endif %}
      {% if visible_classes and "class" not in own_page_types %}
:octicon:`container` Classes
----------------------------

         {% for obj_item in visible_classes %}
{{ obj_item.render(is_own_section = True, containing_class = obj.short_name, current_module = current_module) }}
         {% endfor %}
      {% endif %}
      {% if visible_methods and "method" not in own_page_types %}
:octicon:`cpu` Methods
----------------------

         {% for obj_item in visible_methods %}
{{ obj_item.render(is_own_section = True, containing_class = obj.short_name, current_module = current_module) }}
         {% endfor %}
      {% endif %}
   {% endif %}
{% endif %}
