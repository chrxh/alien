<?php
    const MAX_MINUTES_FOR_INACTIVITY = 60; 
    const PUBLIC_WORKSPACE_TYPE = 0;
    const ALIEN_PROJECT_WORKSPACE_TYPE = 1;
    const PRIVATE_WORKSPACE_TYPE = 2;

    function connectToDB()
    {
        $db = new mysqli("localhost", "[user name]", "[password]", "[database]");
        if ($db->connect_error) {
            exit("Connection error: " . mysqli_connect_error());
        }
        return $db;
    }

    function checkPw($db, $userName, $pw)
    {
        $success = false;

        if ($response = $db->query(
            "SELECT 
                u.PW_HASH as pwHash,
                u.SALT as salt,
                u.ACTIVATION_CODE as activationCode
            FROM user u
            WHERE u.NAME='".addslashes($userName)."'")) {

            $obj = $response->fetch_object();
            if (!$obj) {
                return false;
            }
            $pwHash = hash("sha256", $pw . $obj->salt);

            return $pwHash == $obj->pwHash && $obj->activationCode == "";
        }
        return false;
    }

    function checkPwAndActivationCode($db, $userName, $pw, $activationCode)
    {
        $success = false;

        if ($response = $db->query(
            "SELECT 
                u.PW_HASH as pwHash,
                u.SALT as salt,
                u.ACTIVATION_CODE as activationCode
            FROM user u
            WHERE u.NAME='".addslashes($userName)."'")) {

            $obj = $response->fetch_object();
            if (!$obj) {
                return false;
            }
            $pwHash = hash("sha256", $pw . $obj->salt);

            return $pwHash == $obj->pwHash && $obj->activationCode == $activationCode;
        }
        return false;
    }
?>