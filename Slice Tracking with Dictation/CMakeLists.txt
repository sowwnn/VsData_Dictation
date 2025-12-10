
#-----------------------------------------------------------------------------
set(MODULE_NAME Slice_Tracking_With_Dictation)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  Tracker.py
  Libs/__init__.py
  Libs/LogicLib.py
  Libs/WidgetLib.py
  Utils/__init__.py
  Utils/SliceProcess.py
  Utils/PlotGenerator.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Register the unittest subclass in the main script as a ctest.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()