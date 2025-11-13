import React from "react";
import { motion } from "framer-motion";
import { Activity, Truck, Hospital, Brain } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function AmbulanceSimulationUI() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-100 to-blue-200 flex flex-col items-center justify-center p-6">
      <motion.h1
        className="text-4xl font-bold text-indigo-800 mb-6 tracking-tight drop-shadow-sm"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Smart Ambulance Dispatch Simulation
      </motion.h1>

      <motion.div
        className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl w-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        {/* Patient Card */}
        <Card className="bg-white/40 backdrop-blur-md shadow-md border-none hover:shadow-lg transition-all">
          <CardContent className="p-6 text-center">
            <Activity className="w-12 h-12 mx-auto text-rose-600 mb-4" />
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Patient Alert
            </h2>
            <p className="text-gray-600">
              A patient appears in Zone 3 - Severity: <strong>Critical</strong>
            </p>
          </CardContent>
        </Card>

        {/* Ambulance Card */}
        <Card className="bg-white/40 backdrop-blur-md shadow-md border-none hover:shadow-lg transition-all">
          <CardContent className="p-6 text-center">
            <Truck className="w-12 h-12 mx-auto text-emerald-600 mb-4" />
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Ambulance Status
            </h2>
            <p className="text-gray-600">
              Nearest ambulance is <strong>2 km away</strong>.
            </p>
          </CardContent>
        </Card>

        {/* Hospital Card */}
        <Card className="bg-white/40 backdrop-blur-md shadow-md border-none hover:shadow-lg transition-all">
          <CardContent className="p-6 text-center">
            <Hospital className="w-12 h-12 mx-auto text-blue-600 mb-4" />
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Hospital Availability
            </h2>
            <p className="text-gray-600">
              ICU Beds: <strong>Available</strong> | Specialty: <strong>Cardio</strong>
            </p>
          </CardContent>
        </Card>
      </motion.div>

      {/* Control Buttons */}
      <motion.div
        className="mt-8 flex gap-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <Button className="bg-indigo-700 hover:bg-indigo-800 text-white rounded-xl px-6 py-2 transition-all">
          Start Simulation
        </Button>
        <Button className="bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl px-6 py-2 transition-all">
          Next Episode
        </Button>
      </motion.div>

      {/* Q-Learning Status */}
      <motion.div
        className="mt-10 text-center text-gray-700"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <Brain className="w-8 h-8 inline-block text-indigo-700 mr-2" />
        <span className="font-medium">Agent Learning Progress: 85%</span>
      </motion.div>
    </div>
  );
}
