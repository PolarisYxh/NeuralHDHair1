#-******************************************************************************
#
# Copyright (c) 2012,
#  Sony Pictures Imageworks Inc. and
#  Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# *       Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# *       Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
# *       Neither the name of Sony Pictures Imageworks, nor
# Industrial Light & Magic, nor the names of their contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#-******************************************************************************

# 安装alembic和cask库，参考http://172.20.200.191:8003/pages/viewpage.action?pageId=976774130
from imath import *
import imathnumpy
from alembic.Abc import *
# import unittest
from imath import *
from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
# def setArray( iTPTraits, *iList ):
#     array = iTPTraits.arrayType( len( iList ) )
#     for i in range( len( iList ) ):
#         array[i] = iList[i]
#     return array 
kVertexScope = GeometryScope.kVertexScope
kCubic = CurveType.kCubic
kNonPeriodic = CurvePeriodicity.kNonPeriodic


class CurvesAbc:
    def numpy2array(self, strands):
        self.numVerts = Int32TPTraits.arrayType( len(strands) )
        v_len = 0
        for i in range( len( strands ) ):
            self.numVerts[i] = len(strands[i])
            v_len += len(strands[i])
        self.verts = V3fTPTraits.arrayType( v_len )
        self.uvs = V2fTPTraits.arrayType(v_len)
        self.widths = Float32TPTraits.arrayType(v_len)
        j=0
        for strand in strands:
            for i in range( len( strand ) ):
                self.verts[j] = V3f(strand[i][0],strand[i][1],strand[i][2])
                self.uvs[j] = V2f(0.0,0.0)
                self.widths[j] = 0.2
                j+=1
        # return numVerts,verts,uvs,widths
    def numpy2array1(self, segments, strands):
        self.numVerts = Int32TPTraits.arrayType( len(segments) )
        v_len = strands.shape[0]
        for i,l in enumerate(segments):
            self.numVerts[i] = int(l)
        self.verts = V3fTPTraits.arrayType( v_len )
        self.uvs = V2fTPTraits.arrayType(v_len)
        self.widths = Float32TPTraits.arrayType(v_len)        
        for i in range( strands.shape[0] ):
            self.verts[i] = V3f(strands[i].tolist())
            self.uvs[i] = V2f(0.0,0.0)
            self.widths[i] = 0.2
        # return numVerts,verts,uvs,widths
    def doSample( self, iCurves ):
        curves = iCurves.getSchema()
        widthSamp = OFloatGeomParamSample( self.widths, kVertexScope )
        uvSamp = OV2fGeomParamSample( self.uvs, kVertexScope )
        curvesSamp = OCurvesSchemaSample( self.verts, self.numVerts, kCubic, kNonPeriodic,
                                           widthSamp, uvSamp )
        # knots = curvesSamp.getKnots()
        # # self.assertEquals(len(knots), 0)

        # newKnots = FloatArray(4)
        # for ii in range(4):
        #     newKnots[ii] = ii
        # curvesSamp.setKnots(newKnots)

        # knots = curvesSamp.getKnots()
        # # for ii in range(4):
        # #     self.assertEqual(knots[ii], ii)

        # orders = curvesSamp.getOrders()
        # self.assertEqual(len(orders), 0)

        # newOrder = UnsignedCharArray(3)
        # for ii in range(3):
        #     newOrder[ii] = ii
        # curvesSamp.setOrders(newOrder)

        # orders = curvesSamp.getOrders()
        # for ii in range(3):
        #     self.assertEqual(newOrder[ii], ii)

        curves.set( curvesSamp )

    def CurvesExport(self, strands, outpath):
        """write an oarchive with a curve in it"""
        self.numpy2array(strands)
        myCurves = OCurves( OArchive( outpath ).getTop(),
                            'really_long_curves_name' )
        self.doSample( myCurves )
    def CurvesExport1(self, segments, strands, outpath):
        """write an oarchive with a curve in it"""
        self.numpy2array1(segments,strands)
        myCurves = OCurves( OArchive( outpath ).getTop(),
                            'really_long_curves_name' )
        self.doSample( myCurves )
        
    def testCurvesImport(self,file_name):
        """read an iarchive with a curve in it"""

        myCurves = ICurves( IArchive( file_name ).getTop(),
                            'really_long_curves_name' )
        # myCurves = OCurves( OArchive( file_name ).getTop(),
        #                     'short_hair_splineDescription' )
        # c1 =myCurves.getSchema()
        # c2=c1.getArbGeomParams()
        curves = myCurves.getSchema()

        curvesSamp = curves.getValue()

        positions = curvesSamp.getPositions()

        points = imathnumpy.arrayToNumpy(positions)
        seg = curvesSamp.getCurvesNumVertices()
        segments = imathnumpy.arrayToNumpy(seg)
        numCurves = curvesSamp.getNumCurves()
        return segments,points