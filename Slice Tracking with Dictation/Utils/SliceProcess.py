import vtk
import slicer

def getSliceInfo(viewName='Red'):
    """Get slice information for a specific view."""
    # Map view names to their typical orientations
    viewOrientations = {
        'Red': 'Axial',
        'Yellow': 'Sagittal',
        'Green': 'Coronal'
    }
    
    sliceWidget = slicer.app.layoutManager().sliceWidget(viewName)
    if not sliceWidget:
        return None
        
    sliceNode = sliceWidget.mrmlSliceNode()
    sliceLogic = sliceWidget.sliceLogic()
    volumeNode = sliceLogic.GetBackgroundLayer().GetVolumeNode()
    
    return {
        'sliceNode': sliceNode,
        'sliceLogic': sliceLogic,
        'volumeNode': volumeNode,
        'orientation': viewOrientations.get(viewName, 'Unknown')
    }

def getSliceIndexFromOffset(offset, viewName='Red'):
    """Convert slice offset to slice index for a specific view."""
    # Get the slice information
    sliceInfo = getSliceInfo(viewName)
    if not sliceInfo or not sliceInfo['volumeNode']:
        return 0
    
    sliceNode = sliceInfo['sliceNode']
    volumeNode = sliceInfo['volumeNode']
    
    # Get volume properties
    imageData = volumeNode.GetImageData()
    if not imageData:
        return 0
    
    # Get slice orientation and origin
    sliceToRAS = sliceNode.GetSliceToRAS()
    
    # Create a transform to convert RAS coordinates to IJK (volume) coordinates
    rasToIJK = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIJK)
    
    # Calculate the RAS point at current slice offset
    sliceNormal = [sliceToRAS.GetElement(0, 2), sliceToRAS.GetElement(1, 2), sliceToRAS.GetElement(2, 2)]
    sliceOrigin = [sliceToRAS.GetElement(0, 3), sliceToRAS.GetElement(1, 3), sliceToRAS.GetElement(2, 3)]
    
    # Calculate point at current offset
    point = [
        sliceOrigin[0] + sliceNormal[0] * offset,
        sliceOrigin[1] + sliceNormal[1] * offset,
        sliceOrigin[2] + sliceNormal[2] * offset,
        1.0  # Homogeneous coordinate
    ]
    
    # Transform to IJK space
    ijkPoint = [0, 0, 0, 0]
    rasToIJK.MultiplyPoint(point, ijkPoint)
    
    # The index depends on view orientation
    if sliceInfo['orientation'] == 'Axial':
        sliceIndex = int(round(ijkPoint[2]))
    elif sliceInfo['orientation'] == 'Sagittal':
        sliceIndex = int(round(ijkPoint[0]))
    elif sliceInfo['orientation'] == 'Coronal':
        sliceIndex = int(round(ijkPoint[1]))
    else:
        sliceIndex = int(round(ijkPoint[2]))  # Default to axial
    
    # Don't clamp negative values, but ensure upper bound is respected
    dimensions = imageData.GetDimensions()
    dimIndex = {'Axial': 2, 'Sagittal': 0, 'Coronal': 1}.get(sliceInfo['orientation'], 2)
    if sliceIndex >= dimensions[dimIndex]:
        sliceIndex = dimensions[dimIndex] - 1
    
    return sliceIndex

